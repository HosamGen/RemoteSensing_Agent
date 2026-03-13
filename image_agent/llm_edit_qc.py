import os
import json
import base64
from typing import Dict, Any, Optional, List

import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")


EDIT_QC_SYSTEM_PROMPT = """
You are a quality-control assistant for HIGH-RESOLUTION satellite imagery edits.

You will receive:
1) The original satellite image.
2) The edited satellite image (after inpainting / generative editing).
3) A mask image aligned to the same scene:
   - White/bright = intended edit region
   - Black/dark/transparent = should remain unchanged
4) The user's edit prompt (what the generation was supposed to do).

Your job:
- Determine whether the edit matches the prompt.
- Determine whether changes stayed mostly inside the mask.
- Detect common errors: wrong object edited, edits spilling outside mask, missing/weak edits,
  unrealistic artifacts, boundary blending issues, texture inconsistencies, geometry distortions.

IMPORTANT RULES:
- Base your judgment ONLY on what is visible in the provided images.
- Do not assume the edit succeeded: verify it by comparing original vs edited.
- Penalize changes outside the mask unless the prompt explicitly implies global changes.
- If the intended edit is unclear from the prompt, say so and suggest a clearer prompt.

Return ONLY a JSON object with exactly these keys:

{
  "prompt": "string",
  "verdict": "good" | "ok" | "bad",

  "overall_score": float,                 // 0.0 to 1.0
  "prompt_adherence_score": float,        // 0.0 to 1.0
  "mask_localization_score": float,       // 0.0 to 1.0 (1 = changes confined to mask)
  "background_preservation_score": float, // 0.0 to 1.0 (1 = outside-mask preserved)
  "visual_quality_score": float,          // 0.0 to 1.0 (realism, blending, no artifacts)

  "observed_change_summary": "string",    // what visibly changed
  "prompt_mismatch_summary": "string",    // what's wrong vs prompt (or empty if none)

  "mistakes": [
    {
      "type": "spill_outside_mask | wrong_object | missing_change | artifact | boundary_issue | geometry_issue | texture_mismatch | other",
      "severity": "low" | "medium" | "high",
      "description": "string",
      "location_hint": "string (e.g., 'top right', 'near center', 'along road edge')"
    }
  ],

  "prompt_improvement_tips": ["string", "..."],

  "revised_prompt": "string"
}

Scoring guidance:
- 0.80–1.00 => "good"
- 0.50–0.79 => "ok"
- 0.00–0.49 => "bad"
"""


def _encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def evaluate_highres_edit(
    original_image_path: str,
    edited_image_path: str,
    mask_path: str,
    edit_prompt: str,
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uses GPT-4o (vision) to evaluate whether a masked inpainting edit matches the prompt.

    Parameters
    ----------
    original_image_path : str
        Path to original image.
    edited_image_path : str
        Path to edited image.
    mask_path : str
        Path to mask image (white=edit region).
    edit_prompt : str
        The prompt you used for the generative edit.
    extra_context : Optional[str]
        Optional context to help evaluation (NOT sensor/resolution). Example:
        "Mask is intended to cover only the highway surface."

    Returns
    -------
    Dict[str, Any]
        Parsed JSON result from the model.
    """
    orig_b64 = _encode_image_to_base64(original_image_path)
    edited_b64 = _encode_image_to_base64(edited_image_path)
    mask_b64 = _encode_image_to_base64(mask_path)

    user_text = f"Edit prompt: {edit_prompt}"
    if extra_context:
        user_text += f"\nExtra context: {extra_context}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Chat Completions supports multimodal message content parts including image_url with base64. :contentReference[oaicite:2]{index=2}
    payload = {
        "model": MODEL_NAME,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "messages": [
            {"role": "system", "content": EDIT_QC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_b64}"}},
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    try:
        logger.debug("OpenAI raw response status=%s text=%s", resp.status_code, resp.text)

        resp.raise_for_status()
    except Exception:
        logger.exception(
            "OpenAI API request failed: status=%s text=%s",
            getattr(resp, "status_code", None),
            getattr(resp, "text", None),
        )
        raise

    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)
