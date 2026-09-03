"""
tests/test_grounded_reasons.py

Grounding guarantees for the Gemini weekly-plan prompt (backend/llm_service.py):

- Candidate EVIDENCE is included verbatim in the Gemini request
- The prompt labels EVIDENCE as the ONLY fact source and forbids inventing
  ratings / preferences / popularity / previous meals / nutrition etc.
- Weak or generic evidence triggers a neutral-explanation instruction
- A valid Gemini response whose reasons restate supplied evidence still passes
  backend validation
- Malformed responses still raise LLMServiceError (deterministic fallback path)
- Candidate-only dish validation remains enforced

Gemini is fully mocked — no real API calls.
"""

import os
import sys
import json

from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest  # noqa: E402

from backend.llm_service import _build_prompt, generate_weekly_plan, LLMServiceError  # noqa: E402
from backend.main import _validate_llm_plan, VALID_DAYS  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

def _candidates(n=8):
    """Mirror the recommender's candidate shape with realistic evidence lines."""
    return [
        {
            "dish_id": 101 + i,
            "dish_name": f"Dish {i}",
            "recommendation_score": round(0.90 - 0.05 * i, 4),
            "evidence": (
                ["high family rating", "matches preferences of similar family members"]
                if i < 4
                else ["adds variety to the weekly rotation"]  # weak/generic evidence
            ),
        }
        for i in range(n)
    ]


def _family_context():
    return {"family_name": "Test Family", "members": ["Priya", "Arjun"], "recent_dishes": ["Dosa"]}


def _grounded_plan(dish_ids=None):
    """7-entry plan whose reasons only restate the supplied evidence phrases."""
    ids = dish_ids if dish_ids is not None else [101, 102, 103, 104, 105, 106, 107]
    reasons = [
        "High family rating.",
        "High family rating and matches similar family members.",
        "High family rating.",
        "Matches similar family members.",
        "Adds variety to the weekly rotation.",
        "High family rating.",
        "Adds variety to the weekly rotation.",
    ]
    return [
        {"day": d, "dish_id": did, "reason": reasons[i % len(reasons)]}
        for i, (d, did) in enumerate(zip(VALID_DAYS, ids))
    ]


def _mock_gemini_returning(raw_text):
    """Return (mock_client, mock_genai) wired so Gemini responds with raw_text."""
    mock_response = MagicMock()
    mock_response.text = raw_text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client
    return mock_client, mock_genai


def _call_gemini(mock_genai):
    """Run generate_weekly_plan inside the standard mocked-google environment."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
        with patch.dict(sys.modules, {
            "google": MagicMock(genai=mock_genai),
            "google.genai": mock_genai,
            "google.genai.types": MagicMock(),
        }):
            return generate_weekly_plan(_candidates(), _family_context())


def _last_prompt(mock_client):
    """The prompt string Gemini actually received."""
    return mock_client.models.generate_content.call_args.kwargs["contents"]


# --------------------------------------------------------------------------- #
# A. Candidate evidence is actually included in the Gemini request
# --------------------------------------------------------------------------- #

def test_prompt_includes_each_candidates_evidence():
    prompt = _build_prompt(_candidates(), _family_context())
    # evidence phrases appear verbatim
    assert "high family rating" in prompt
    assert "matches preferences of similar family members" in prompt
    assert "adds variety to the weekly rotation" in prompt
    # evidence is attached to the right candidates with an explicit label
    assert "EVIDENCE:" in prompt
    assert "dish_id=101" in prompt and 'name="Dish 0"' in prompt
    assert "recommendation_score=0.900" in prompt


def test_evidence_reaches_the_actual_gemini_request():
    client, genai = _mock_gemini_returning(json.dumps(_grounded_plan()))
    result = _call_gemini(genai)
    prompt = _last_prompt(client)
    assert len(result) == 7
    assert "EVIDENCE:" in prompt
    assert "high family rating" in prompt
    assert "adds variety to the weekly rotation" in prompt


# --------------------------------------------------------------------------- #
# B. The prompt explicitly forbids inventing evidence
# --------------------------------------------------------------------------- #

def test_prompt_labels_evidence_as_only_fact_source():
    prompt = _build_prompt(_candidates(), _family_context())
    assert "EVIDENCE line is the ONLY source of facts" in prompt
    assert "Never invent dishes or dish IDs" in prompt


def test_prompt_forbids_inventing_family_facts():
    prompt = _build_prompt(_candidates(), _family_context())
    for banned in (
        "NEVER invent",
        "no exact ratings",
        "vote counts",
        "member preferences",
        "previous meals",
        "nutrition",
    ):
        assert banned in prompt, f"missing anti-invention clause: {banned}"


def test_prompt_weak_evidence_asks_for_neutral_reason():
    # All candidates carry only the generic variety evidence.
    weak = [dict(c, evidence=["adds variety to the weekly rotation"]) for c in _candidates()]
    prompt = _build_prompt(weak, _family_context())
    assert "weak or generic" in prompt
    assert "neutral, honest reason" in prompt
    assert "variety/balance" in prompt


# --------------------------------------------------------------------------- #
# C. A valid Gemini response with grounded reasons still passes
# --------------------------------------------------------------------------- #

def test_grounded_reasons_response_passes_validation():
    client, genai = _mock_gemini_returning(json.dumps(_grounded_plan()))
    result = _call_gemini(genai)
    assert _last_prompt(client)  # request happened

    candidate_ids = {c["dish_id"] for c in _candidates()}
    group_dish_ids = set(range(100, 200))
    validated = _validate_llm_plan(result, candidate_ids, group_dish_ids)
    assert len(validated) == 7
    assert {e["day"] for e in validated} == set(VALID_DAYS)
    for e in validated:
        assert len(e["reason"]) >= 5
        assert e["dish_id"] in candidate_ids


# --------------------------------------------------------------------------- #
# D. Malformed responses still raise (caller falls back deterministically)
# --------------------------------------------------------------------------- #

def test_malformed_gemini_response_still_raises():
    client, genai = _mock_gemini_returning("this is not JSON")
    with pytest.raises(LLMServiceError, match="not valid JSON"):
        _call_gemini(genai)


# --------------------------------------------------------------------------- #
# E. Candidate-only dish validation remains enforced
# --------------------------------------------------------------------------- #

def test_candidate_only_dish_ids_still_enforced():
    # dish 150 is in the group but was never sent to Gemini as a candidate
    plan = _grounded_plan(dish_ids=[101, 102, 103, 104, 105, 106, 150])
    candidate_ids = {c["dish_id"] for c in _candidates()}   # 101..108
    group_dish_ids = set(range(100, 200))
    with pytest.raises(ValueError, match="candidate set"):
        _validate_llm_plan(plan, candidate_ids, group_dish_ids)
