# image_agent/external_tools.py
"""
External tool wrappers.
Simplified: LISAt and FLUX only.
SAM, SAM3, Gemini, and BetaRisk have been removed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .subprocess_utils import run_cmd, extract_first_json


# ----------------------------
# Config dataclasses
# ----------------------------

@dataclass
class LISAtConfig:
    env_name: str = "lisat"
    env_runner: str = "conda"
    script_path: str = "../LISAt_code/infer_lisat.py"
    model_path: str = "checkpoints/LISAt-7b"
    image_size: int = 1024
    max_new_tokens: int = 512


@dataclass
class FluxConfig:
    env_name: str = "flux"
    env_runner: str = "conda"
    output_dir: str = "../flux/output"
    guidance: float = 20.0
    num_steps: int = 30
    offload: bool = True


# ----------------------------
# LISAt
# ----------------------------

def lisat_segment_mask(
    image_path: str,
    prompt: str,
    out_mask_path: str,
    cfg: LISAtConfig,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Run LISAt segmentation via subprocess.
    Returns parsed JSON metadata from the script.
    """
    cmd = [
        "python", cfg.script_path,
        "--image", image_path,
        "--prompt", prompt,
        "--out_mask", out_mask_path,
        "--model_path", cfg.model_path,
        "--image_size", str(cfg.image_size),
        "--max_new_tokens", str(max_new_tokens),
    ]

    result = run_cmd(
        cmd,
        env_name=cfg.env_name,
        env_runner=cfg.env_runner,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LISAt failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    try:
        meta = extract_first_json(result.stdout)
    except Exception:
        meta = {"stdout_tail": result.stdout[-500:]}

    return meta


# ----------------------------
# FLUX
# ----------------------------

def flux_inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    out_path: str,
    cfg: FluxConfig,
) -> Dict[str, Any]:
    """
    Run FLUX inpainting via subprocess.
    Returns parsed JSON metadata from the script.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "flux_fill_inpaint",
        "--image", image_path,
        "--mask", mask_path,
        "--prompt", prompt,
        "--out", out_path,
        "--guidance", str(cfg.guidance),
        "--num_steps", str(cfg.num_steps),
        "--output_dir", cfg.output_dir,
    ]
    if cfg.offload:
        cmd.append("--offload")

    result = run_cmd(
        cmd,
        env_name=cfg.env_name,
        env_runner=cfg.env_runner,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FLUX failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr[-1000:]}"
        )

    try:
        meta = extract_first_json(result.stdout)
    except Exception:
        meta = {"stdout_tail": result.stdout[-500:]}

    return meta
