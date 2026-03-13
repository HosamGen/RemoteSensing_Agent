# image_agent/highres_mask_qc.py

import os
import json
import base64
from typing import Dict, Any, Optional

import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")

MASK_MATCH_SYSTEM_PROMPT = """
You are a quality-control assistant for segmentation of HIGH-RESOLUTION satellite imagery.

You will receive:
- A high-resolution satellite image (top-down).
- A binary or soft mask image aligned to the same scene
  (white/bright = predicted object, black/dark/transparent = background).
- The intended object type, e.g. "highway", "road network", "building", "parking lot".

Assume the imagery is high resolution: individual roads, lanes, buildings, and other structures
are clearly visible.

Your job is to judge how well the mask isolates the intended object and to describe any errors.

Consider:

- COVERAGE:
  - Does the mask include the full visible extent of the object?
  - Are there obvious gaps where the object continues but the mask disappears?

- PRECISION:
  - Does the mask avoid covering clearly wrong areas (e.g., fields, buildings, water)?
  - Are there large regions of mask that do NOT correspond to the object type?

- SHAPE & ALIGNMENT:
  - Does the mask follow the correct geometry of the object?
    *For a highway*, it should follow the linear, multi-lane structure of the road network.
  - Is it correctly aligned with the visual highway or is it shifted?

- FRAGMENTATION:
  - Is the object continuous but the mask is broken into many pieces?
  - Or is the mask a single coherent region where appropriate?

You must return ONLY a JSON object with this structure:

{
  "object_type": "string",                  // echo the intended object type
  "overall_score": float,                   // 0.0 to 1.0
  "coverage_score": float,                  // 0.0 to 1.0
  "precision_score": float,                 // 0.0 to 1.0
  "fragmentation_score": float,             // 0.0 to 1.0, 1 = single coherent region
  "verdict": "good" | "ok" | "bad",
  "issues": [
    "short bullet-style descriptions of major problems"
  ],
  "summary": "1-3 sentence explanation of your judgement"
}

Scoring guidance (heuristic, not exact IoU):

- overall_score 0.8 – 1.0:
  - "good": mask closely matches the object; only minor errors.
- overall_score 0.5 – 0.8:
  - "ok": mask roughly matches but with noticeable under/over-segmentation.
- overall_score 0.0 – 0.5:
  - "bad": mask largely misses the object or is mostly on the wrong areas.

If the mask clearly does not correspond to the intended object at all, use very low scores
(e.g., 0.0 – 0.2) and verdict "bad", and explain why.
"""


def _encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def evaluate_mask_match(
    image_path: str,
    mask_path: str,
    object_type: str,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate how well a mask matches an intended object in a HIGH-RES satellite image.

    Parameters
    ----------
    image_path : str
        Path to the original satellite image.
    mask_path : str
        Path to the predicted mask image (same alignment as image_path).
    object_type : str
        Intended object type, e.g. "highway".
    extra_context : str, optional
        Optional extra info such as "main highway running north-south".

    Returns
    -------
    Dict[str, Any]
        Parsed JSON with keys:
          - object_type
          - overall_score
          - coverage_score
          - precision_score
          - fragmentation_score
          - verdict
          - issues (list)
          - summary
    """
    img_b64 = _encode_image_to_base64(image_path)
    mask_b64 = _encode_image_to_base64(mask_path)

    if extra_context:
        user_text = (
            f"The intended object is: {object_type}. "
            f"Extra context: {extra_context}."
        )
    else:
        user_text = f"The intended object is: {object_type}."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": MASK_MATCH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            user_text
                            + " Judge how well the mask matches the object in the image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{mask_b64}",
                        },
                    },
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    try:
        # Always emit a debug-level copy of the response text so developers can see it in logs.
        logger.debug("OpenAI raw response status=%s text=%s", resp.status_code, resp.text)

        resp.raise_for_status()
    except Exception:
        # When an error occurs, include the response body in the exception log for easier debugging.
        logger.exception(
            "OpenAI API request failed: status=%s text=%s",
            getattr(resp, "status_code", None),
            getattr(resp, "text", None),
        )
        raise

    content = resp.json()["choices"][0]["message"]["content"]

    return json.loads(content)
