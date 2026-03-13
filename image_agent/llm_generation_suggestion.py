# image_agent/llm_generation_suggestion.py
import base64
import json
import mimetypes
import os
from typing import Any, Dict, Optional, List

import requests
import logging
import time

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_S = float(os.environ.get("OPENAI_TIMEOUT_S", "60"))

## Removed this from the prompt to avoid the FLUX model from generating an actual "mask" image.
#    The prompt must always end with context like:
#    "Only edit inside the mask region; keep everything outside unchanged."

GEN_SUGGEST_SYSTEM_PROMPT = """
You are an assistant that proposes realistic, localized edits to HIGH-RESOLUTION satellite imagery
for inpainting with a diffusion model (FLUX Fill).

Inputs:
- Original satellite image.
- A binary mask aligned to the image (white/bright = region allowed to change; black = must remain unchanged).
- A user goal (e.g. "make this safer", "lower the risk of accidents", "increase the risk score").
- Information about what the mask region represents (e.g., road, building, open land, parking lot).
- OPTIONAL: Previous failed attempts with their QC scores and issues.

Your task:
1) CRITICAL: Examine the mask to understand what is inside the WHITE region.
   - Is it a road/highway? A building? Open land? Water? Parking lot? Intersection?
   - The edit MUST be appropriate for what's actually there.

2) Interpret the user's goal:
   - goal_direction: "decrease_risk" (make safer) or "increase_risk" (adversarial stress-testing).

3) IF PREVIOUS FAILURES ARE PROVIDED:
   - Review what was already tried and failed.
   - Suggest COMPLETELY DIFFERENT approaches that avoid those specific problems.

4) Propose 3-5 candidate edits that are:
   - Plausible from a top-down satellite view.
   - Constrained to the masked region ONLY.
   - Realistic (no text overlays, no labels, no watermarks, no surreal elements).
   - MEANINGFULLY DIFFERENT from each other.
   - DIFFERENT from previous failures if provided.

5) PROMPT FORMAT — CRITICAL:
   Each edit_prompt MUST follow this structure:
   [Subject description] + [Scene/setting context] + [Style/material cues] + [Lighting] + [Camera/technical]

   The prompt describes WHAT THE MASKED REGION SHOULD LOOK LIKE after the edit.
   It describes the final appearance, NOT an action ("replace X with Y" or "add something").

   GOOD prompt examples:
     "Single-family homes with driveways and small yards, suburban residential neighborhood,
      overhead satellite view, natural daylight, sharp top-down aerial photograph."

     "Top-down satellite view of a harbor filled with cargo ships and container vessels,
      industrial port setting, natural daylight, high-resolution aerial image."

     "Satellite view of agricultural terraces with irregular plots, bright soil contrast,
      and irrigation channels, top-down perspective, midday natural lighting."

     "Dense urban grid of medium-rise apartment blocks with rooftop AC units and fire escapes,
      overhead city view, overcast natural light, photorealistic satellite photograph."

     "Concrete highway with clear lane markings and a wide median barrier,
      top-down aerial perspective, bright daylight, high-resolution satellite imagery."

   BAD prompt examples (do NOT use these patterns):
     "Replace the parking lot with residential buildings." — action verb, not a description
     "Add median barriers to the road." — action verb
     "Make this safer by adding sidewalks." — action verb + goal language
     "Improve the area." — vague, no visual description

SAFETY CONSTRAINT:
- If goal is "increase_risk", treat as synthetic robustness testing only.
  Keep suggestions at the level of visual complexity cues (denser markings, more lanes,
  more intersections within the mask). Do NOT advise real-world harm.

Output a single JSON object:

{
  "goal_text": "string",
  "goal_direction": "decrease_risk|increase_risk",
  "goal_interpretation": "string (1-2 sentences)",
  "detected_mask_content": "string (road/building/open_land/water/parking/intersection/etc)",
  "detected_mask_content_explanation": "string (brief explanation of what you see)",
  "learning_from_failures": "string (if previous_failures provided: what you learned)",
  "candidate_edits": [
    {
      "id": "edit_1",
      "title": "short name",
      "rationale": "why this likely moves risk score AND is appropriate for this mask content",
      "edit_prompt": "Subject + Scene + Style + Lighting + Camera description. Only edit inside the mask region; keep everything outside unchanged.",
      "diversity_note": "how this approach differs from other candidates"
    }
  ],
  "recommended_edit_id": "edit_#",
  "recommended_prompt": "string (must equal the corresponding candidate edit_prompt)",
  "notes": "any caveats or assumptions"
}

IMPORTANT:
- Always fill in "detected_mask_content" with your assessment.
- Each candidate must have an edit_prompt following the Subject+Scene+Style+Lighting+Camera format.
- Candidates must be meaningfully different approaches, not minor variations.
- If previous failures provided, new suggestions must avoid those specific issues.

Return only valid JSON with double quotes.
""".strip()


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def _encode_b64(path: str) -> str:
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large (>8MB): {path} ({size} bytes). Please resize/compress.")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def suggest_generation_prompt(
    image_path: str,
    mask_path: str,
    goal_text: str,
    target_object_type: Optional[str] = None,
    region_description: Optional[str] = None,
    global_description: Optional[str] = None,
    current_risk_mean: Optional[float] = None,
    previous_failures: Optional[List[Dict[str, Any]]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    temperature: float = 0.2,
    image_detail: str = "auto",
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    img_b64 = _encode_b64(image_path)
    mask_b64 = _encode_b64(mask_path)

    context_bits: List[str] = [f"User goal: {goal_text}"]
    if current_risk_mean is not None:
        context_bits.append(f"Current risk_mean (proxy): {current_risk_mean:.3f}")
    if global_description:
        context_bits.append(f"Scene description: {global_description}")
    if region_description:
        context_bits.append(f"Mask region description: {region_description}")
    if target_object_type:
        context_bits.append(f"Target object type (expected): {target_object_type}")

    if previous_failures and len(previous_failures) > 0:
        context_bits.append("\n**PREVIOUS FAILED ATTEMPTS (please avoid repeating these):**")
        for i, failure in enumerate(previous_failures, 1):
            context_bits.append(f"Attempt {i}:")
            context_bits.append(f"  Prompt: {failure.get('prompt', 'N/A')}")
            context_bits.append(f"  QC Score: {failure.get('qc_score', 0.0):.2f}")
            if failure.get("issues"):
                context_bits.append(f"  Issues: {', '.join(failure['issues'][:3])}")
        context_bits.append(
            "\nPlease suggest COMPLETELY DIFFERENT approaches that avoid these specific problems."
        )

    user_text = (
        "\n".join(context_bits)
        + "\n\nExamine the mask (white region) and propose candidate edits using the"
        + " Subject+Scene+Style+Lighting+Camera prompt format."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": GEN_SUGGEST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": image_detail}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mask_b64}", "detail": image_detail}},
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=timeout_s)

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
