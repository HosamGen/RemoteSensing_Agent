# image_agent/segmentation_agent.py
"""
Segmentation Agent: Handles all mask generation logic.
Simplified: LISAt only (SAM, SAM3, and bbox-crop fallbacks removed).
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .external_tools import (
    LISAtConfig,
    lisat_segment_mask,
)
from .image_utils import load_image_size
from .llm_mask_qc import evaluate_mask_match


# ----------------------------
# IoU deduplication helper
# ----------------------------

def _mask_iou(path_a: str, path_b: str) -> float:
    """Compute IoU between two binary masks. Returns float in [0,1]."""
    try:
        a = _load_mask_bool(path_a)
        b = _load_mask_bool(path_b)
        if a.shape != b.shape:
            return 0.0
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _deduplicate_masks(masks: List[Dict[str, Any]], iou_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate masks (IoU >= iou_threshold).
    Among duplicates, keep the one with the highest qc score.
    """
    if not masks:
        return masks
    sorted_masks = sorted(masks, key=lambda m: m["mask_qc"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for candidate in sorted_masks:
        duplicate = False
        for keeper in kept:
            if _mask_iou(candidate["mask_path"], keeper["mask_path"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _load_mask_bool(mask_path: str) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    arr = np.array(m)
    return arr > 0


def mask_qc_score(mask_path: str) -> float:
    """Cheap heuristic score for masks to pre-filter candidates (not the final QC)."""
    mask = _load_mask_bool(mask_path)
    area = float(mask.mean())

    if area < 0.0005:
        return 0.0
    if area > 0.65:
        return 0.1

    peak = 0.05
    lo, hi = 0.005, 0.25
    if area < lo:
        return max(0.0, area / lo) * 0.6
    if area > hi:
        return max(0.0, 1.0 - (area - hi) / (0.65 - hi)) * 0.6

    score = math.exp(-abs(area - peak) / 0.06)
    return float(min(1.0, max(0.0, score)))


# ----------------------------
# Mask smoothing
# ----------------------------

def _resolve_smooth_script(smooth_script_path: str) -> Optional[Path]:
    """Find smooth_mask.py in image_agent/ or project root."""
    p = Path(smooth_script_path)
    if p.is_file():
        return p

    agent_dir = Path(__file__).resolve().parent
    p2 = agent_dir / "smooth_mask.py"
    if p2.is_file():
        return p2

    root = agent_dir.parent
    p3 = root / "smooth_mask.py"
    if p3.is_file():
        return p3

    return None


def _smooth_mask(
    in_mask_path: str,
    out_mask_path: str,
    smooth_config: Dict[str, Any],
    log: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Calls smooth_mask.py as a subprocess."""
    import subprocess
    import sys

    script = _resolve_smooth_script(smooth_config.get("script_path", "smooth_mask.py"))
    if not script:
        log.info("smooth_mask.py not found; skipping smoothing.")
        return None

    cmd = [
        sys.executable,
        str(script),
        "--in", in_mask_path,
        "--out", out_mask_path,
        "--thresh", "127",
        "--min-area", str(int(smooth_config.get("min_area", 200))),
        "--close-radius", str(int(smooth_config.get("close_radius", 3))),
        "--widen-px", str(int(smooth_config.get("widen_px", 0))),
    ]
    if smooth_config.get("fill_holes", True):
        cmd.append("--fill-holes")
    if smooth_config.get("complete_road", False):
        cmd.append("--complete-road")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            log.info("smooth_mask failed rc=%d stderr=%s", proc.returncode, proc.stderr[-500:])
            return None
        return {
            "cmd": cmd,
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
        }
    except Exception as e:
        log.info("smooth_mask exception: %s", e)
        return None


# ----------------------------
# Main segmentation agent
# ----------------------------

class SegmentationAgent:
    """
    Handles all segmentation logic for a single region.
    Simplified: LISAt only. SAM, SAM3 and bbox-crop fallback removed.

    Key behaviours:
    - Runs LISAt once (it's deterministic).
    - Collects all masks that pass QC threshold.
    - Returns a list of accepted masks in `accepted_masks`, plus the single best in `mask_path`.
    """

    def __init__(
        self,
        image_path: str,
        region: Dict[str, Any],
        seg_prompts: List[str],  # Kept for compatibility
        lisat_cfg: LISAtConfig,
        smoothing_config: Dict[str, Any],
        qc_threshold: float,
        max_attempts_per_region: int,
        roi_pad_pct: float,
        prompts_dir: Path,
        seg_dir: Path,
        log: logging.Logger,
    ):
        self.image_path = image_path
        self.region = region
        self.region_id = str(region.get("id", "unknown"))
        self.object_type = str(region.get("feature_type", "other"))
        self.region_desc = str(region.get("description", ""))

        self.center_x = int(region.get("point_x", 512))
        self.center_y = int(region.get("point_y", 512))

        self.seg_prompts = seg_prompts
        self.lisat_cfg = lisat_cfg
        self.smoothing_config = smoothing_config
        self.qc_threshold = qc_threshold
        self.max_attempts = max_attempts_per_region
        self.roi_pad_pct = roi_pad_pct

        self.prompts_dir = prompts_dir
        self.seg_dir = seg_dir
        self.log = log

        self.smooth_coverage_threshold = smoothing_config.get("coverage_threshold", 0.6)
        self.smooth_fragmentation_threshold = smoothing_config.get("fragmentation_threshold", 0.6)

        # Results
        self.accepted_masks: List[Dict[str, Any]] = []
        self.best_mask_path: Optional[str] = None
        self.best_mask_qc = -1.0
        self.best_mask_source = ""
        self.best_mask_qc_obj: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def run_llm_mask_qc(self, mask_path: str, tag: str, extra: str) -> Tuple[float, Dict[str, Any]]:
        """Run LLM-based mask QC and save to disk."""
        try:
            qc_obj = evaluate_mask_match(
                image_path=self.image_path,
                mask_path=mask_path,
                object_type=self.object_type,
                extra_context=extra,
            )
        except Exception as e:
            qc_obj = {
                "error": str(e),
                "overall_score": 0.0,
                "verdict": "bad",
                "issues": ["LLM mask QC call failed"],
            }

        from .full_pipeline import _write_json
        qc_dir = Path(os.environ.get("IMAGE_AGENT_QC_DIR", "qc"))
        qc_file = qc_dir / f"mask_qc_{self.region_id}_{tag}.json"
        _write_json(qc_file, qc_obj)

        return float(qc_obj.get("overall_score", 0.0)), qc_obj

    def try_smoothing(self, mask_path: str, sidx: int, source_tag: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Try smoothing a mask. Only smooths if QC indicates it would help.
        Returns (best_path, best_qc, best_qc_obj).
        """
        orig_qc, orig_qc_obj = self.run_llm_mask_qc(
            mask_path=mask_path,
            tag=source_tag,
            extra=f"source={source_tag} | target={self.object_type} | region={self.region_desc}",
        )

        verdict = orig_qc_obj.get("verdict", "bad")
        coverage_score = orig_qc_obj.get("coverage_score", 1.0)
        fragmentation_score = orig_qc_obj.get("fragmentation_score", 1.0)

        should_smooth = (
            verdict != "bad" and
            (coverage_score <= self.smooth_coverage_threshold or
             fragmentation_score <= self.smooth_fragmentation_threshold)
        )

        if not should_smooth or not self.smoothing_config.get("enabled", True):
            self.log.info("Skipping smoothing for %s (verdict=%s, coverage=%.2f, frag=%.2f)",
                         source_tag, verdict, coverage_score, fragmentation_score)
            return mask_path, orig_qc, orig_qc_obj

        self.log.info("Applying smoothing to %s (coverage=%.2f, fragmentation=%.2f)",
                     source_tag, coverage_score, fragmentation_score)
        smooth_path = str(self.seg_dir / f"{self.region_id}_{source_tag}_smooth.png")
        sm_meta = _smooth_mask(mask_path, smooth_path, self.smoothing_config, self.log)

        if sm_meta and Path(smooth_path).is_file():
            sm_qc, sm_qc_obj = self.run_llm_mask_qc(
                mask_path=smooth_path,
                tag=f"{source_tag}_smooth",
                extra=f"source={source_tag}+smoothing | target={self.object_type} | region={self.region_desc}",
            )

            self.events.append({
                "event": "mask_smoothed",
                "reason": f"coverage={coverage_score:.2f}, fragmentation={fragmentation_score:.2f}",
                "original_mask": mask_path,
                "smoothed_mask": smooth_path,
                "original_qc": orig_qc,
                "smoothed_qc": sm_qc,
            })

            if sm_qc > orig_qc:
                self.log.info("Smoothing improved QC: %.3f -> %.3f", orig_qc, sm_qc)
                return smooth_path, sm_qc, sm_qc_obj
            else:
                self.log.info("Smoothing did not improve QC, keeping original")

        return mask_path, orig_qc, orig_qc_obj

    def _record_accepted(self, mask_path: str, qc_score: float, qc_obj: Dict[str, Any], source: str) -> None:
        """Record a mask that passed QC. Updates best_mask if this is the highest QC."""
        entry = {
            "mask_path": mask_path,
            "mask_qc": qc_score,
            "mask_source": source,
            "mask_qc_obj": qc_obj,
        }
        self.accepted_masks.append(entry)
        self.log.info("Seg[%s] accepted mask #%d from source=%s qc=%.3f",
                      self.region_id, len(self.accepted_masks), source, qc_score)

        if qc_score > self.best_mask_qc:
            self.best_mask_qc = qc_score
            self.best_mask_path = mask_path
            self.best_mask_source = source
            self.best_mask_qc_obj = qc_obj

    def segment_region(self) -> Dict[str, Any]:
        """
        Run segmentation for this region using LISAt only.
        LISAt is deterministic so it is called once. The result is passed through
        optional smoothing and LLM QC.
        """
        # LISAt is deterministic — run once only
        self._try_lisat(sidx=0)

        # IoU deduplication (in case smoothed + unsmoothed both passed)
        if len(self.accepted_masks) > 1:
            before = len(self.accepted_masks)
            self.accepted_masks = _deduplicate_masks(self.accepted_masks, iou_threshold=0.85)
            removed = before - len(self.accepted_masks)
            if removed:
                self.log.info(
                    "Seg[%s] IoU dedup removed %d near-duplicate mask(s); %d remain",
                    self.region_id, removed, len(self.accepted_masks),
                )

        if not self.accepted_masks:
            self.log.info("Region %s: no acceptable masks found (best qc=%.3f)", self.region_id, self.best_mask_qc)
            return {
                "success": False,
                "region_id": self.region_id,
                "best_qc": self.best_mask_qc,
                "accepted_masks": [],
                "events": self.events,
            }

        self.log.info("Region %s: found %d accepted mask(s), best qc=%.3f",
                      self.region_id, len(self.accepted_masks), self.best_mask_qc)
        return {
            "success": True,
            "region_id": self.region_id,
            "region": self.region,
            "mask_path": self.best_mask_path,
            "mask_qc": self.best_mask_qc,
            "mask_source": self.best_mask_source,
            "mask_qc_obj": self.best_mask_qc_obj,
            "accepted_masks": self.accepted_masks,
            "events": self.events,
        }

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _try_lisat(self, sidx: int) -> bool:
        """Try LISAt segmentation. Returns True if a good mask was found."""
        out_mask_path = str(self.seg_dir / f"{self.region_id}_lisat_mask_{sidx:02d}.png")

        planner_prompt = self.region.get("lisat_segmentation_prompt", "")
        if not planner_prompt:
            planner_prompt = f"Segment the {self.object_type} in this image."

        from .full_pipeline import _write_json
        _write_json(self.prompts_dir / f"lisat_input_{self.region_id}_{sidx:02d}.json", {
            "model": "lisat",
            "region_id": self.region_id,
            "attempt": sidx,
            "object_type": self.object_type,
            "region_description": self.region_desc,
            "final_prompt_sent": planner_prompt,
            "image_path": self.image_path,
            "timestamp": datetime.now().isoformat(),
        })

        self.log.info("Seg[%s] LISAt | prompt=%s", self.region_id, planner_prompt[:100])

        try:
            seg_meta = lisat_segment_mask(
                image_path=self.image_path,
                prompt=planner_prompt,
                out_mask_path=out_mask_path,
                cfg=self.lisat_cfg,
                max_new_tokens=self.lisat_cfg.max_new_tokens,
            )
        except Exception as e:
            self.events.append({"event": "seg_lisat_failed", "try": sidx, "err": str(e)})
            return False

        final_path, qc_score, qc_obj = self.try_smoothing(
            out_mask_path, sidx, f"lisat_{sidx:02d}"
        )

        self.events.append({
            "event": "seg_lisat",
            "try": sidx,
            "mask": final_path,
            "qc": qc_score,
            "meta": seg_meta,
        })

        if qc_score > self.best_mask_qc:
            self.best_mask_qc = qc_score
            self.best_mask_path = final_path
            self.best_mask_source = f"lisat_{sidx:02d}" + ("_smooth" if final_path.endswith("_smooth.png") else "")
            self.best_mask_qc_obj = qc_obj

        if qc_score >= self.qc_threshold:
            source = f"lisat_{sidx:02d}" + ("_smooth" if final_path.endswith("_smooth.png") else "")
            self._record_accepted(final_path, qc_score, qc_obj, source)
            return True

        return False
