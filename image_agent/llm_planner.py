# image_agent/llm_planner.py
"""
Planning agent: interprets the user goal and produces a list of regions to edit.
Simplified: SAM3 keyword, use_bbox, and bbox_px fields removed from region schema.
"""
from __future__ import annotations

import json
import os
import base64
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


PLANNER_SYSTEM_PROMPT = """
You are a planning agent for a satellite image editing pipeline.

Given:
- A high-resolution satellite image (top-down view).
- A user goal (e.g. "make this safer", "reduce road risk", "increase greenery").

Your job:
1. Identify 1-5 distinct regions in the image that should be edited to achieve the goal.
2. For each region, produce a LISAt segmentation prompt and a short description.

Each region must include:
- id: unique string (e.g. "region_1")
- feature_type: object type as a short noun phrase (e.g. "road", "highway", "parking lot", "building")
- description: 1-2 sentence description of what the region is and where it is
- point_x, point_y: approximate pixel coordinates (integers) of the region's center
- lisat_segmentation_prompt: a natural language instruction for LISAt, e.g.
    "Please segment the main highway running through the center of this image."

Return ONLY a JSON object:

{
  "goal_text": "string",
  "goal_direction": "decrease_risk | increase_risk | other",
  "global_scene_description": "string (brief description of the whole image)",
  "regions": [
    {
      "id": "region_1",
      "feature_type": "string",
      "description": "string",
      "point_x": int,
      "point_y": int,
      "lisat_segmentation_prompt": "string"
    }
  ],
  "notes": "string (optional caveats)"
}

Rules:
- Only identify regions that are clearly visible and relevant to the goal.
- Prefer regions with the highest potential impact on the goal.
- Keep feature_type as a simple noun (road, highway, building, parking lot, intersection).
- lisat_segmentation_prompt must be a full sentence starting with "Please segment...".
- Return only valid JSON with double quotes.
""".strip()


def _encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def plan_regions(
    image_path: str,
    goal_text: str,
    prompt_history: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    Call the planning LLM to identify regions to edit.

    Parameters
    ----------
    image_path : str
        Path to the satellite image.
    goal_text : str
        User goal, e.g. "make this safer".
    prompt_history : list, optional
        Previous planning outputs (for multi-epoch runs).
    temperature : float
        LLM temperature (default 0 for deterministic planning).
    model : str
        OpenAI model name.

    Returns
    -------
    Dict with keys: goal_text, goal_direction, global_scene_description, regions, notes.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    img_b64 = _encode_b64(image_path)

    context_bits = [f"User goal: {goal_text}"]

    if prompt_history:
        context_bits.append("\n**PREVIOUS PLANNING OUTPUTS (for reference):**")
        for i, prev in enumerate(prompt_history[-3:], 1):  # last 3 entries
            context_bits.append(
                f"Epoch {i}: edited {len(prev.get('regions', []))} region(s). "
                f"Scene: {prev.get('global_scene_description', '')[:100]}"
            )
        context_bits.append("Identify the same or new regions as appropriate.")

    user_text = "\n".join(context_bits)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
    try:
        logger.debug("Planner raw response status=%s text=%s", resp.status_code, resp.text)
        resp.raise_for_status()
    except Exception:
        logger.exception(
            "Planner API request failed: status=%s text=%s",
            getattr(resp, "status_code", None),
            getattr(resp, "text", None),
        )
        raise

    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)
