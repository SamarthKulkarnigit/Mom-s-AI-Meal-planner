"""
backend/llm_service.py

Gemini LLM service for AI-assisted weekly meal plan generation.

Responsibilities:
- Build a grounded prompt from ML-generated candidates + family context
- Call Gemini using google-genai SDK with structured JSON output
- Return a parsed list of DayPlan dicts, or raise LLMServiceError

NEVER used directly for database writes.
The calling backend function (main.py) owns persistence and validation.

Environment variables:
  GEMINI_API_KEY    – required for AI path; if absent, raises LLMServiceError
  GEMINI_MODEL      – optional, default "gemini-1.5-flash" (free-tier friendly)
  GEMINI_TIMEOUT_MS – optional request timeout in milliseconds (default 60000)
"""

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TIMEOUT_MS = 60_000  # bounded request so hangs trigger the caller's fallback


def _timeout_ms() -> int:
    try:
        return int(os.getenv("GEMINI_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)).strip())
    except ValueError:
        return DEFAULT_TIMEOUT_MS


class LLMServiceError(Exception):
    """Raised for all recoverable LLM failures — caller should use fallback."""
    pass


def _build_prompt(candidates: List[Dict], family_context: Dict) -> str:
    """Build a compact, grounded prompt. No invented data."""
    family_name = family_context.get("family_name", "the family")
    members = family_context.get("members", [])
    recent_dishes = family_context.get("recent_dishes", [])

    members_str = ", ".join(members) if members else "Unknown"
    recent_str = ", ".join(recent_dishes) if recent_dishes else "None"

    cands_lines = []
    for c in candidates:
        ev = "; ".join(c.get("evidence", [])) or "No specific signals"
        cands_lines.append(
            f'  - dish_id={c["dish_id"]}, name="{c["dish_name"]}", '
            f'score={c["recommendation_score"]:.3f}, signals=[{ev}]'
        )
    cands_block = "\n".join(cands_lines)

    prompt = (
        f'You are a meal planner assistant for a family called "{family_name}".\n'
        f"Family members: {members_str}\n"
        f"Recent meals (avoid repeating these if possible): {recent_str}\n\n"
        f"Your task: Select exactly 7 dishes from the candidates below — one per day — "
        f"for a 7-day weekly meal plan.\n"
        f"Days must be: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday "
        f"(exactly one each, in this order).\n\n"
        f"STRICT RULES:\n"
        f"- You MUST only use dish_id values from the candidate list below. Do NOT invent dish IDs.\n"
        f"- Do NOT repeat a dish across multiple days.\n"
        f'- Each "reason" must be 1-2 sentences grounded in the signals provided '
        f"(rating, poll votes, variety). Do NOT mention nutrition, ingredients, or "
        f"health benefits not shown in the signals.\n"
        f'- Return ONLY a JSON array of exactly 7 objects with keys: "day", "dish_id", "reason".\n\n'
        f"Candidates:\n{cands_block}\n\n"
        f"Return JSON only. No markdown fences. No explanation outside the JSON.\n"
        f'Format:\n'
        f'[\n'
        f'  {{"day": "Monday", "dish_id": <integer>, "reason": "<grounded reason>"}},\n'
        f'  ...\n'
        f']\n'
    )
    return prompt


def generate_weekly_plan(
    candidates: List[Dict[str, Any]],
    family_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Call Gemini to select/organize a 7-day meal plan from the candidate list.

    Returns a list of 7 dicts: [{"day": "Monday", "dish_id": 123, "reason": "..."}, ...]

    Raises LLMServiceError on any failure (missing key, API error, bad response).
    The caller in main.py is responsible for fallback handling.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMServiceError("GEMINI_API_KEY is not set")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise LLMServiceError("google-genai package is not installed")

    prompt = _build_prompt(candidates, family_context)
    logger.info("LLM: calling Gemini model=%s with %d candidates", model_name, len(candidates))

    try:
        # Bounded timeout so API hangs (rate limits, network stalls) surface as
        # LLMServiceError and trigger the deterministic fallback downstream.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_timeout_ms()),
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise LLMServiceError(f"Gemini API call failed: {exc}") from exc

    raw_text = ""
    try:
        raw_text = response.text
    except Exception:
        raise LLMServiceError("Gemini returned an empty or unparseable response")

    if not raw_text or not raw_text.strip():
        raise LLMServiceError("Gemini returned an empty response body")

    # Strip accidental markdown fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMServiceError(
            f"Gemini response is not valid JSON: {exc}. Raw: {cleaned[:300]}"
        ) from exc

    if not isinstance(parsed, list):
        raise LLMServiceError(f"Expected JSON array, got {type(parsed).__name__}")

    logger.info("LLM: received %d plan entries from Gemini", len(parsed))
    return parsed
