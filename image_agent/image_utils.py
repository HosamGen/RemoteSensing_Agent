from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from PIL import Image


@dataclass(frozen=True)
class BBoxPx:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def load_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size  # (w, h)


# DEPRECATED: pct_bbox_to_px is no longer used
# Planner now returns absolute pixel coordinates (point_x, point_y) instead of percentage bboxes
# This function is kept for backward compatibility with generation_agent's ROI cropping
# but is not used in segmentation_agent anymore.
#
# def pct_bbox_to_px(bbox_pct: Dict[str, float], img_w: int, img_h: int, pad_pct: float = 0.0) -> BBoxPx:
#     ...


def crop_image(image_path: str, bbox: BBoxPx, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        crop = im.crop((bbox.left, bbox.top, bbox.right, bbox.bottom))
        crop.save(out_path)
    return out_path


def paste_crop_mask_into_full(
    full_image_path: str,
    crop_mask_path: str,
    bbox: BBoxPx,
    out_mask_path: str,
) -> str:
    """
    Create a full-size mask image (L mode) where the crop_mask is pasted into bbox.
    Assumes crop_mask is aligned to the crop region size.
    """
    Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)

    with Image.open(full_image_path) as im:
        full_w, full_h = im.size

    full_mask = Image.new("L", (full_w, full_h), color=0)
    with Image.open(crop_mask_path) as cm:
        cm_l = cm.convert("L")
        # If LISAt returns 0/255 already, good. Otherwise we keep as-is.
        # Ensure mask matches bbox size; if not, resize to bbox (best-effort).
        if cm_l.size != (bbox.width, bbox.height):
            cm_l = cm_l.resize((bbox.width, bbox.height))
        full_mask.paste(cm_l, (bbox.left, bbox.top))

    full_mask.save(out_mask_path)
    return out_mask_path
