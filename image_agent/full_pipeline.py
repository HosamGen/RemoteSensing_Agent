# image_agent/full_pipeline.py
"""
Full pipeline: orchestrates planner → segmentation (LISAt) → generation (FLUX) → QC.
Simplified: BetaRisk, SAM, SAM3, and Gemini removed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .external_tools import LISAtConfig, FluxConfig, flux_inpaint
from .llm_planner import plan_regions
from .llm_generation_suggestion import suggest_generation_prompt
from .llm_edit_qc import evaluate_highres_edit
from .segmentation_agent import SegmentationAgent

logger = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------

@dataclass
class AgentConfig:
    output_root: str = "outputs/full_runs"

    mask_qc_threshold: float = 0.7
    edit_qc_threshold: float = 0.7
    max_mask_attempts: int = 10
    max_edit_attempts: int = 10
    roi_pad_pct: float = 0.15

    lisat: LISAtConfig = field(default_factory=LISAtConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)

    # Smoothing
    enable_mask_smoothing: bool = True
    smooth_script_path: str = "image_agent/smooth_mask.py"
    smooth_min_area: int = 200
    smooth_close_radius: int = 3
    smooth_widen_px: int = 0
    smooth_fill_holes: bool = True
    smooth_complete_road: bool = False
    smooth_coverage_threshold: float = 0.6
    smooth_fragmentation_threshold: float = 0.6

    # Per-region budgets
    max_mask_attempts_per_region: int = 3
    max_edit_attempts_per_region: int = 3

    # Full image mode
    use_full_image_for_generation: bool = False


# ----------------------------
# Helpers
# ----------------------------

def _write_json(path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def _setup_run_dir(output_root: str, image_path: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(image_path).stem
    run_dir = Path(output_root) / f"{stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_smoothing_config(cfg: AgentConfig) -> Dict[str, Any]:
    return {
        "enabled": cfg.enable_mask_smoothing,
        "script_path": cfg.smooth_script_path,
        "min_area": cfg.smooth_min_area,
        "close_radius": cfg.smooth_close_radius,
        "widen_px": cfg.smooth_widen_px,
        "fill_holes": cfg.smooth_fill_holes,
        "complete_road": cfg.smooth_complete_road,
        "coverage_threshold": cfg.smooth_coverage_threshold,
        "fragmentation_threshold": cfg.smooth_fragmentation_threshold,
    }


def _crop_roi(image_path: str, mask_path: str, roi_pad_pct: float, out_path: str) -> Optional[str]:
    """Crop image to bounding box of mask with padding. Returns out_path or None on failure."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        arr = np.array(mask) > 0

        rows = np.any(arr, axis=1)
        cols = np.any(arr, axis=0)
        if not rows.any():
            return None

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        h, w = arr.shape
        pad_y = int((r1 - r0) * roi_pad_pct)
        pad_x = int((c1 - c0) * roi_pad_pct)

        r0 = max(0, r0 - pad_y)
        r1 = min(h, r1 + pad_y)
        c0 = max(0, c0 - pad_x)
        c1 = min(w, c1 + pad_x)

        cropped = img.crop((c0, r0, c1, r1))
        cropped.save(out_path)
        return out_path
    except Exception as e:
        logger.warning("ROI crop failed: %s", e)
        return None


# ----------------------------
# Main pipeline
# ----------------------------

def run_full_agent(
    image_path: str,
    goal_text: str,
    cfg: AgentConfig,
) -> Dict[str, Any]:
    """
    Run the full pipeline:
      1. Plan regions (LLM planner)
      2. Segment each region (LISAt)
      3. Generate edits (FLUX)
      4. QC edits (LLM judge)

    Returns a result dict with run metadata and all successful edits.
    """
    run_dir = _setup_run_dir(cfg.output_root, image_path)
    os.environ["IMAGE_AGENT_QC_DIR"] = str(run_dir / "qc")

    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("full_pipeline")
    log.info("Run dir: %s", run_dir)
    log.info("Image: %s | Goal: %s", image_path, goal_text)

    # Copy input image into run dir for reference
    input_copy = run_dir / ("input" + Path(image_path).suffix)
    shutil.copy2(image_path, input_copy)

    smoothing_config = _build_smoothing_config(cfg)

    # Sub-directories
    seg_dir = run_dir / "segmentation"
    gen_dir = run_dir / "generation"
    prompts_dir = run_dir / "prompts"
    qc_dir = run_dir / "qc"
    for d in [seg_dir, gen_dir, prompts_dir, qc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_successful_edits: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: Plan
    # ------------------------------------------------------------------
    log.info("=== Step 1: Planning ===")
    try:
        plan = plan_regions(image_path=image_path, goal_text=goal_text)
    except Exception as e:
        log.error("Planning failed: %s", e)
        result = {"success": False, "error": str(e), "run_dir": str(run_dir)}
        _write_json(run_dir / "result.json", result)
        return result

    _write_json(prompts_dir / "plan.json", plan)
    regions = plan.get("regions", [])
    log.info("Planner identified %d region(s)", len(regions))

    if not regions:
        result = {
            "success": False,
            "error": "Planner returned no regions",
            "run_dir": str(run_dir),
            "plan": plan,
        }
        _write_json(run_dir / "result.json", result)
        return result

    # ------------------------------------------------------------------
    # Step 2: Segment + Generate per region
    # ------------------------------------------------------------------
    region_results: List[Dict[str, Any]] = []

    for region in regions:
        region_id = str(region.get("id", "unknown"))
        log.info("=== Region %s: %s ===", region_id, region.get("feature_type", ""))

        r_seg_dir = seg_dir / region_id
        r_seg_dir.mkdir(parents=True, exist_ok=True)

        # -- Segmentation --
        seg_agent = SegmentationAgent(
            image_path=image_path,
            region=region,
            seg_prompts=[],  # unused in simplified version
            lisat_cfg=cfg.lisat,
            smoothing_config=smoothing_config,
            qc_threshold=cfg.mask_qc_threshold,
            max_attempts_per_region=cfg.max_mask_attempts_per_region,
            roi_pad_pct=cfg.roi_pad_pct,
            prompts_dir=prompts_dir,
            seg_dir=r_seg_dir,
            log=log,
        )

        seg_result = seg_agent.segment_region()
        _write_json(r_seg_dir / "seg_result.json", seg_result)

        if not seg_result["success"]:
            log.warning("Region %s: segmentation failed (best qc=%.3f)", region_id, seg_result.get("best_qc", 0))
            region_results.append({"region_id": region_id, "success": False, "stage": "segmentation"})
            continue

        accepted_masks = seg_result["accepted_masks"]
        log.info("Region %s: %d accepted mask(s)", region_id, len(accepted_masks))

        # -- Generate edits for each accepted mask --
        region_edits: List[Dict[str, Any]] = []

        for mask_entry in accepted_masks:
            mask_path = mask_entry["mask_path"]
            mask_qc = mask_entry["mask_qc"]
            mask_source = mask_entry["mask_source"]

            # Choose image for generation (full image or ROI crop)
            if cfg.use_full_image_for_generation:
                gen_image_path = image_path
            else:
                roi_path = str(r_seg_dir / f"{region_id}_{mask_source}_roi.png")
                gen_image_path = _crop_roi(image_path, mask_path, cfg.roi_pad_pct, roi_path) or image_path

            # Get generation suggestion
            try:
                suggestion = suggest_generation_prompt(
                    image_path=gen_image_path,
                    mask_path=mask_path,
                    goal_text=goal_text,
                    target_object_type=region.get("feature_type"),
                    region_description=region.get("description"),
                    global_description=plan.get("global_scene_description"),
                )
                _write_json(prompts_dir / f"suggestion_{region_id}_{mask_source}.json", suggestion)
            except Exception as e:
                log.warning("Region %s: suggestion failed: %s", region_id, e)
                continue

            edit_prompt = suggestion.get("recommended_prompt", "")
            if not edit_prompt:
                log.warning("Region %s: no recommended_prompt in suggestion", region_id)
                continue

            # Run FLUX edits (up to max_edit_attempts_per_region)
            failed_attempts: List[Dict[str, Any]] = []

            for edit_idx in range(cfg.max_edit_attempts_per_region):
                out_edit_path = str(gen_dir / f"{region_id}_{mask_source}_edit_{edit_idx:02d}.png")

                try:
                    flux_meta = flux_inpaint(
                        image_path=gen_image_path,
                        mask_path=mask_path,
                        prompt=edit_prompt,
                        out_path=out_edit_path,
                        cfg=cfg.flux,
                    )
                except Exception as e:
                    log.warning("Region %s edit %d: FLUX failed: %s", region_id, edit_idx, e)
                    failed_attempts.append({"prompt": edit_prompt, "qc_score": 0.0, "issues": [str(e)]})
                    break  # FLUX errors are not retried differently

                if not Path(out_edit_path).is_file():
                    log.warning("Region %s edit %d: FLUX produced no output file", region_id, edit_idx)
                    break

                # QC
                try:
                    qc_result = evaluate_highres_edit(
                        original_image_path=gen_image_path,
                        edited_image_path=out_edit_path,
                        mask_path=mask_path,
                        edit_prompt=edit_prompt,
                    )
                    _write_json(qc_dir / f"edit_qc_{region_id}_{mask_source}_{edit_idx:02d}.json", qc_result)
                except Exception as e:
                    log.warning("Region %s edit %d: QC failed: %s", region_id, edit_idx, e)
                    qc_result = {"overall_score": 0.0, "verdict": "bad", "issues": [str(e)]}

                edit_qc = float(qc_result.get("overall_score", 0.0))
                log.info("Region %s edit %d: qc=%.3f verdict=%s", region_id, edit_idx, edit_qc, qc_result.get("verdict"))

                if edit_qc >= cfg.edit_qc_threshold:
                    edit_record = {
                        "region_id": region_id,
                        "mask_path": mask_path,
                        "mask_qc": mask_qc,
                        "mask_source": mask_source,
                        "edited_image_path": out_edit_path,
                        "edit_prompt": edit_prompt,
                        "edit_qc": edit_qc,
                        "edit_qc_obj": qc_result,
                    }
                    region_edits.append(edit_record)
                    all_successful_edits.append(edit_record)
                    log.info("Region %s: accepted edit (qc=%.3f)", region_id, edit_qc)
                    break  # One good edit per mask is enough
                else:
                    failed_attempts.append({
                        "prompt": edit_prompt,
                        "qc_score": edit_qc,
                        "issues": [m.get("description", "") for m in qc_result.get("mistakes", [])],
                    })

                    # Re-suggest with failure context if we have attempts left
                    if edit_idx + 1 < cfg.max_edit_attempts_per_region:
                        try:
                            suggestion = suggest_generation_prompt(
                                image_path=gen_image_path,
                                mask_path=mask_path,
                                goal_text=goal_text,
                                target_object_type=region.get("feature_type"),
                                region_description=region.get("description"),
                                global_description=plan.get("global_scene_description"),
                                previous_failures=failed_attempts,
                            )
                            edit_prompt = suggestion.get("recommended_prompt", edit_prompt)
                        except Exception as e:
                            log.warning("Region %s: re-suggestion failed: %s", region_id, e)

        region_results.append({
            "region_id": region_id,
            "success": len(region_edits) > 0,
            "num_edits": len(region_edits),
            "edits": region_edits,
        })

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    result = {
        "success": len(all_successful_edits) > 0,
        "run_dir": str(run_dir),
        "image_path": image_path,
        "goal_text": goal_text,
        "num_regions": len(regions),
        "num_successful_edits": len(all_successful_edits),
        "all_successful_edits": all_successful_edits,
        "region_results": region_results,
        "plan": plan,
    }
    _write_json(run_dir / "result.json", result)
    log.info("Done. %d successful edit(s) across %d region(s).", len(all_successful_edits), len(regions))
    return result
