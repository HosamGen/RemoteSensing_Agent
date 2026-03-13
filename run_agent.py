import argparse
import json
from pathlib import Path

from image_agent.full_pipeline import AgentConfig, run_full_agent
from image_agent.external_tools import LISAtConfig, FluxConfig


def build_cfg_from_args(args) -> AgentConfig:
    cfg = AgentConfig()

    cfg.output_root = args.output_root
    cfg.mask_qc_threshold = args.mask_qc_threshold
    cfg.edit_qc_threshold = args.edit_qc_threshold
    cfg.max_mask_attempts = args.max_mask_attempts
    cfg.max_edit_attempts = args.max_edit_attempts
    cfg.roi_pad_pct = args.roi_pad_pct

    cfg.lisat = LISAtConfig(
        env_name=args.lisat_env,
        env_runner=args.env_runner,
        script_path=args.lisat_script,
        model_path=args.lisat_model_path,
        image_size=args.lisat_image_size,
        max_new_tokens=args.lisat_max_new_tokens,
    )

    cfg.flux = FluxConfig(
        env_name=args.flux_env,
        env_runner=args.env_runner,
        output_dir=args.flux_output_dir,
        guidance=args.flux_guidance,
        num_steps=args.flux_num_steps,
        offload=not args.no_flux_offload,
    )

    # Smoothing config
    cfg.enable_mask_smoothing = not args.disable_mask_smoothing
    cfg.smooth_script_path = args.smooth_script_path
    cfg.smooth_min_area = args.smooth_min_area
    cfg.smooth_close_radius = args.smooth_close_radius
    cfg.smooth_widen_px = args.smooth_widen_px
    cfg.smooth_fill_holes = not args.no_smooth_fill_holes
    cfg.smooth_complete_road = args.smooth_complete_road
    cfg.smooth_coverage_threshold = args.smooth_coverage_threshold
    cfg.smooth_fragmentation_threshold = args.smooth_fragmentation_threshold

    # Per-region budgets
    cfg.max_mask_attempts_per_region = args.max_mask_attempts_per_region
    cfg.max_edit_attempts_per_region = args.max_edit_attempts_per_region

    # Full image mode
    cfg.use_full_image_for_generation = args.use_full_image

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Run satellite image editing agent pipeline (LISAt + FLUX).")
    ap.add_argument("--image", required=True, help="Path to input satellite image")
    ap.add_argument("--goal", required=True, help="User goal text, e.g. 'make this safer'")
    ap.add_argument("--output_root", default="outputs/full_runs")

    ap.add_argument("--env_runner", default="conda", choices=["conda", "mamba"])

    # Thresholds / limits
    ap.add_argument("--mask_qc_threshold", type=float, default=0.7)
    ap.add_argument("--edit_qc_threshold", type=float, default=0.7)
    ap.add_argument("--max_mask_attempts", type=int, default=10)
    ap.add_argument("--max_edit_attempts", type=int, default=10)
    ap.add_argument("--roi_pad_pct", type=float, default=0.15)

    # LISAt
    ap.add_argument("--lisat_env", default="lisat")
    ap.add_argument("--lisat_script", default="../LISAt_code/infer_lisat.py")
    ap.add_argument("--lisat_model_path", default="checkpoints/LISAt-7b")
    ap.add_argument("--lisat_image_size", type=int, default=1024)
    ap.add_argument("--lisat_max_new_tokens", type=int, default=512)

    # FLUX
    ap.add_argument("--flux_env", default="flux")
    ap.add_argument("--flux_output_dir", default="../flux/output")
    ap.add_argument("--flux_guidance", type=float, default=20.0)
    ap.add_argument("--flux_num_steps", type=int, default=30)
    ap.add_argument("--no_flux_offload", action="store_true")

    # Per-region budgets
    ap.add_argument("--max_mask_attempts_per_region", type=int, default=3)
    ap.add_argument("--max_edit_attempts_per_region", type=int, default=3)

    # Full image mode
    ap.add_argument("--use_full_image", action="store_true",
                    help="Use full image for generation instead of ROI cropping")

    # Smoothing
    ap.add_argument("--disable_mask_smoothing", action="store_true")
    ap.add_argument("--smooth_script_path", default="image_agent/smooth_mask.py")
    ap.add_argument("--smooth_min_area", type=int, default=200)
    ap.add_argument("--smooth_close_radius", type=int, default=3)
    ap.add_argument("--smooth_widen_px", type=int, default=0)
    ap.add_argument("--no_smooth_fill_holes", action="store_true")
    ap.add_argument("--smooth_complete_road", action="store_true")
    ap.add_argument("--smooth_coverage_threshold", type=float, default=0.6)
    ap.add_argument("--smooth_fragmentation_threshold", type=float, default=0.6)

    args = ap.parse_args()

    cfg = build_cfg_from_args(args)

    result = run_full_agent(
        image_path=args.image,
        goal_text=args.goal,
        cfg=cfg,
    )

    print(json.dumps(result, indent=2))

    out_dir = Path(result["run_dir"])
    print(f"\nResults saved to: {out_dir / 'result.json'}")

    if result.get("all_successful_edits"):
        print(f"\n{len(result['all_successful_edits'])} successful edits:")
        for i, edit in enumerate(result["all_successful_edits"][:5], 1):
            print(f"  {i}. QC={edit['edit_qc']:.3f}")
        if len(result["all_successful_edits"]) > 5:
            print(f"  ... and {len(result['all_successful_edits']) - 5} more")


if __name__ == "__main__":
    main()
