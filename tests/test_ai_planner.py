"""
tests/test_ai_planner.py

Unit tests for Phase 2 AI planner validation logic and LLM service.

ALL tests mock the Gemini API — no real API calls are made.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import _validate_llm_plan, VALID_DAYS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

CANDIDATE_IDS = {101, 102, 103, 104, 105, 106, 107, 108, 109}
GROUP_DISH_IDS = set(range(100, 150))  # group owns dishes 100-149


def _good_plan(dish_ids=None):
    """Return a valid 7-entry plan using provided dish_ids (defaults to 101-107)."""
    if dish_ids is None:
        dish_ids = [101, 102, 103, 104, 105, 106, 107]
    return [
        {"day": d, "dish_id": did, "reason": "High family rating and good variety."}
        for d, did in zip(VALID_DAYS, dish_ids)
    ]


# ─── 1. Successful plan – passes validation ────────────────────────────────────

def test_valid_plan_passes():
    validated = _validate_llm_plan(_good_plan(), CANDIDATE_IDS, GROUP_DISH_IDS)
    assert len(validated) == 7
    assert {e["day"] for e in validated} == set(VALID_DAYS)


# ─── 2. Exactly 7 unique days ──────────────────────────────────────────────────

def test_wrong_number_of_entries():
    plan = _good_plan()[:5]
    with pytest.raises(ValueError, match="exactly 7"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


def test_duplicate_days():
    plan = _good_plan()
    plan[1]["day"] = plan[0]["day"]  # make Monday appear twice
    with pytest.raises(ValueError, match="Duplicate day"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 3. Invalid dish ID (not an integer) ──────────────────────────────────────

def test_non_integer_dish_id():
    plan = _good_plan()
    plan[0]["dish_id"] = "not-a-number"
    with pytest.raises(ValueError, match="not an integer"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 4. Dish doesn't belong to the family ─────────────────────────────────────

def test_dish_not_in_group():
    plan = _good_plan()
    plan[0]["dish_id"] = 999  # not in GROUP_DISH_IDS
    with pytest.raises(ValueError, match="does not belong to this group"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 5. Dish not in candidate set ─────────────────────────────────────────────

def test_dish_not_in_candidates():
    plan = _good_plan()
    plan[0]["dish_id"] = 120  # in group (100-149) but NOT in CANDIDATE_IDS (101-109)
    with pytest.raises(ValueError, match="candidate set"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 6. Duplicate dishes ──────────────────────────────────────────────────────

def test_duplicate_dish_ids():
    ids = [101, 101, 103, 104, 105, 106, 107]  # 101 repeated
    plan = _good_plan(dish_ids=ids)
    with pytest.raises(ValueError, match="more than once"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 7. Malformed JSON (non-dict entries) ─────────────────────────────────────

def test_non_dict_entry():
    plan = _good_plan()
    plan[2] = "just a string"
    with pytest.raises(ValueError, match="not a dict"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 8. Not a list at all ─────────────────────────────────────────────────────

def test_not_a_list():
    with pytest.raises(ValueError, match="must be a list"):
        _validate_llm_plan({"day": "Monday"}, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 9. Invalid day name ──────────────────────────────────────────────────────

def test_invalid_day_name():
    plan = _good_plan()
    plan[0]["day"] = "Funday"
    with pytest.raises(ValueError, match="invalid day"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 10. Reason too short ─────────────────────────────────────────────────────

def test_empty_reason():
    plan = _good_plan()
    plan[3]["reason"] = "ok"  # less than 5 chars
    with pytest.raises(ValueError, match="reason is too short"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 11. Missing required key ─────────────────────────────────────────────────

def test_missing_key():
    plan = _good_plan()
    del plan[0]["reason"]
    with pytest.raises(ValueError, match="missing keys"):
        _validate_llm_plan(plan, CANDIDATE_IDS, GROUP_DISH_IDS)


# ─── 12. LLM service — missing API key triggers LLMServiceError ───────────────

def test_llm_service_missing_key():
    from backend.llm_service import generate_weekly_plan, LLMServiceError
    candidates = [
        {"dish_id": 101, "dish_name": "Rajma Chawal", "recommendation_score": 0.9, "evidence": ["high rating"]}
    ] * 7
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(LLMServiceError, match="GEMINI_API_KEY"):
            generate_weekly_plan(candidates, {"family_name": "Test"})


# ─── 13. LLM service — API failure raises LLMServiceError ─────────────────────

def test_llm_service_api_failure():
    from backend.llm_service import generate_weekly_plan, LLMServiceError

    candidates = [
        {"dish_id": i, "dish_name": f"Dish {i}", "recommendation_score": 0.5, "evidence": []}
        for i in range(1, 8)
    ]
    family_context = {"family_name": "TestFamily", "members": [], "recent_dishes": []}

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
        with patch("backend.llm_service.genai", create=True) as mock_genai:
            mock_genai.Client.return_value = mock_client
            with pytest.raises(LLMServiceError, match="API call failed"):
                generate_weekly_plan(candidates, family_context)


# ─── 14. LLM service — malformed JSON response ────────────────────────────────

def test_llm_service_invalid_json():
    from backend.llm_service import generate_weekly_plan, LLMServiceError
    import sys

    candidates = [
        {"dish_id": i, "dish_name": f"Dish {i}", "recommendation_score": 0.5, "evidence": []}
        for i in range(1, 8)
    ]
    family_context = {"family_name": "TestFamily", "members": [], "recent_dishes": []}

    mock_response = MagicMock()
    mock_response.text = "This is not JSON at all"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai, "google.genai.types": MagicMock()}):
            with pytest.raises(LLMServiceError, match="not valid JSON"):
                generate_weekly_plan(candidates, family_context)


# ─── 15. LLM service — valid structured JSON response is parsed correctly ─────

def test_llm_service_valid_response():
    from backend.llm_service import generate_weekly_plan
    import sys, json

    candidates = [
        {"dish_id": i, "dish_name": f"Dish {i}", "recommendation_score": 0.5, "evidence": []}
        for i in range(101, 109)
    ]
    family_context = {"family_name": "TestFamily", "members": ["Alice"], "recent_dishes": []}

    expected_json = [
        {"day": d, "dish_id": 101 + i, "reason": "Good family rating."}
        for i, d in enumerate(VALID_DAYS)
    ]

    mock_response = MagicMock()
    mock_response.text = json.dumps(expected_json)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai, "google.genai.types": MagicMock()}):
            result = generate_weekly_plan(candidates, family_context)

    assert len(result) == 7
    assert result[0]["day"] == "Monday"
    assert result[0]["dish_id"] == 101


# ─── 16. GEMINI_MODEL env var is respected ────────────────────────────────────

def test_gemini_model_env_var():
    from backend.llm_service import generate_weekly_plan
    import sys, json

    candidates = [
        {"dish_id": 101 + i, "dish_name": f"Dish {i}", "recommendation_score": 0.5, "evidence": []}
        for i in range(8)
    ]
    family_context = {"family_name": "TestFamily", "members": [], "recent_dishes": []}
    expected = [{"day": d, "dish_id": 101 + i, "reason": "Test reason ok."} for i, d in enumerate(VALID_DAYS)]

    mock_response = MagicMock()
    mock_response.text = json.dumps(expected)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key", "GEMINI_MODEL": "gemini-2.0-flash"}):
        with patch.dict(sys.modules, {"google": MagicMock(genai=mock_genai), "google.genai": mock_genai, "google.genai.types": MagicMock()}):
            generate_weekly_plan(candidates, family_context)
            call_kwargs = mock_client.models.generate_content.call_args
            # model is passed as a positional or keyword arg
            used_model = call_kwargs[1].get("model") or (call_kwargs[0][0] if call_kwargs[0] else None)
            assert used_model == "gemini-2.0-flash"
