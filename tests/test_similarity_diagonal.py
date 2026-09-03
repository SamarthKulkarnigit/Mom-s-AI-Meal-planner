"""
tests/test_similarity_diagonal.py

Regression tests for the self-similarity / diagonal fix in ml_recommender.py:

- A dish's similarity to ITSELF must never contribute to its CF or content score
  (the diagonal of both similarity matrices is zeroed).
- CF can now distinguish a dish that is similar to another rated dish from an
  unrelated / unrated dish.
- Content similarity can now distinguish similar dish names from unrelated ones.
- Single-dish / tiny-catalog cases stay valid (scores 0, never NaN/inf).
- get_candidates_for_group() uses the corrected signals (no self-boost).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402
import ml_recommender as mr  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _make_group(label):
    code = db.generate_group_code()
    s = SessionLocal()
    try:
        s.add(m.Group(group_code=code, family_name=f"{label} Family", creator=label))
        s.commit()
    finally:
        s.close()
    return code


def _add_dishes(code, names):
    s = SessionLocal()
    try:
        for name in names:
            s.add(m.Dish(group_code=code, name=name, source="Poll"))
        s.commit()
    finally:
        s.close()


def _rate(code, dish, rating, user, week=1):
    assert db.rate_dish(code, dish, rating, user_name=user, week=week) is True


# --------------------------------------------------------------------------- #
# 1. CF: own similarity never used; similar-vs-unrelated distinguishability
# --------------------------------------------------------------------------- #

def test_cf_diagonal_is_zero_and_self_similarity_is_never_used():
    ratings = pd.DataFrame({
        "user": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "dish": ["A Dish", "B Dish", "A Dish", "B Dish", "C Dish", "D Dish"],
        "rating": [5.0, 5.0, 4.0, 4.0, 1.0, 2.0],
        "week": [1, 1, 1, 1, 1, 1],
    })
    sim = mr._build_cf_sim(ratings)
    # Diagonal (own similarity) is exactly 0 for every rated dish.
    assert list(np.diag(sim.values)) == [0.0] * len(sim)

    # A dish's score = max similarity to OTHER rated dishes: the row max equals
    # the best OFF-diagonal cross-similarity, never a free 1.0 from itself.
    for dish in sim.index:
        assert sim.loc[dish, dish] == 0.0
        others = sim.loc[dish].drop(dish)
        assert sim.loc[dish].max() == pytest.approx(others.max())
        assert sim.loc[dish].max() <= 1.0 + 1e-9  # allow fp overshoot of identical vectors
        assert np.isfinite(sim.loc[dish].max())


def test_cf_distinguishes_similar_dish_from_unrelated_one():
    # A and B rated identically by all four users -> cross-sim ~1.
    # D is anti-correlated with A/B -> low cross-sim. E never rated -> 0.
    code = _make_group("cf_diag")
    names = ["Alpha Curry", "Beta Curry", "Delta Soup", "Epsilon Salad"]
    _add_dishes(code, names)
    for u, vals in {
        "u1": {"Alpha Curry": 5.0, "Beta Curry": 5.0, "Delta Soup": 1.0},
        "u2": {"Alpha Curry": 5.0, "Beta Curry": 5.0, "Delta Soup": 1.0},
        "u3": {"Alpha Curry": 1.0, "Beta Curry": 1.0, "Delta Soup": 5.0},
        "u4": {"Alpha Curry": 1.0, "Beta Curry": 1.0, "Delta Soup": 5.0},
    }.items():
        for dish, rating in vals.items():
            _rate(code, dish, rating, u)

    sim = mr._build_cf_sim(db.load_data(f"ratings_{code}.csv"))
    # Rated dishes only.
    assert set(sim.index) == {"Alpha Curry", "Beta Curry", "Delta Soup"}
    # Cross-sim inside the correlated pair is near 1; the anti-correlated dish is lower.
    assert sim.loc["Alpha Curry", "Beta Curry"] > 0.9
    assert sim.loc["Alpha Curry", "Delta Soup"] < 0.5
    # Self entries are excluded.
    assert sim.loc["Alpha Curry", "Alpha Curry"] == 0.0
    # Never-rated dish is not even in the matrix (its cf stays 0 downstream).
    assert "Epsilon Salad" not in sim.index


# --------------------------------------------------------------------------- #
# 2. Content: own similarity never used; name-similarity works
# --------------------------------------------------------------------------- #

def test_content_diagonal_is_zero():
    dishes = pd.DataFrame({"dish": ["Paneer Butter Masala", "Palak Paneer", "Idli Sambar"]})
    sim = mr._build_content_sim(dishes)
    assert list(np.diag(sim.values)) == [0.0] * len(sim)
    for dish in sim.index:
        assert sim.loc[dish, dish] == 0.0


def test_content_distinguishes_similar_names_from_unrelated():
    dishes = pd.DataFrame({"dish": ["Paneer Butter Masala", "Palak Paneer", "Idli Sambar"]})
    sim = mr._build_content_sim(dishes)
    # Two paneer dishes share a token -> positive similarity.
    assert sim.loc["Paneer Butter Masala", "Palak Paneer"] > 0.0
    assert sim.loc["Palak Paneer", "Paneer Butter Masala"] > 0.0
    # Idli Sambar shares no tokens with either -> zero similarity.
    assert sim.loc["Paneer Butter Masala", "Idli Sambar"] == 0.0
    # Row max for the related dish comes from the other dish (not itself).
    assert sim.loc["Paneer Butter Masala"].max() == sim.loc["Paneer Butter Masala", "Palak Paneer"]


# --------------------------------------------------------------------------- #
# 3. Single-dish / tiny catalog edge cases
# --------------------------------------------------------------------------- #

def test_single_dish_content_sim_is_zero_and_finite():
    dishes = pd.DataFrame({"dish": ["Only Dish"]})
    sim = mr._build_content_sim(dishes)
    assert sim.shape == (1, 1)
    assert sim.iloc[0, 0] == 0.0
    assert np.isfinite(sim.iloc[0, 0])


def test_single_rated_dish_cf_is_zero_and_finite():
    ratings = pd.DataFrame({
        "user": ["u1", "u2"],
        "dish": ["Lone Dish", "Lone Dish"],
        "rating": [4.0, 5.0],
        "week": [1, 1],
    })
    sim = mr._build_cf_sim(ratings)
    assert sim.shape == (1, 1)
    assert sim.iloc[0, 0] == 0.0  # no OTHER rated dish to be similar to
    assert np.isfinite(sim.iloc[0, 0])


def test_scores_are_finite_and_bounded_in_full_planner():
    code = _make_group("cf_finite")
    names = [f"Meal {i}" for i in range(1, 9)]
    _add_dishes(code, names)
    for u in ("u1", "u2"):
        for name in names[:4]:
            _rate(code, name, 5.0 if name in ("Meal 1", "Meal 2") else 2.0, u)

    plan, stats = mr.generate_weekly_plan_for_group(code, return_stats=True)
    assert plan is not None and len(plan) == 7
    for col in ("rating_score", "popularity_score", "cf_score", "content_score", "hybrid_score", "final_score"):
        vals = pd.to_numeric(stats[col], errors="coerce").dropna()
        assert vals.between(0.0, 1.0).all(), col
        assert np.isfinite(vals).all(), col


# --------------------------------------------------------------------------- #
# 4. get_candidates_for_group() uses the corrected signals
# --------------------------------------------------------------------------- #

def test_get_candidates_cf_evidence_requires_real_similarity():
    code = _make_group("cf_candidates")
    names = ["Alpha Curry", "Beta Curry", "Delta Soup", "Epsilon Salad"]
    _add_dishes(code, names)

    # Alpha & Beta: same ratings from everyone -> highly similar.
    # Delta: anti-correlated with the pair -> rated but NOT similar.
    # Epsilon: never rated at all.
    for u, vals in {
        "u1": {"Alpha Curry": 5.0, "Beta Curry": 5.0, "Delta Soup": 1.0},
        "u2": {"Alpha Curry": 5.0, "Beta Curry": 5.0, "Delta Soup": 1.0},
        "u3": {"Alpha Curry": 1.0, "Beta Curry": 1.0, "Delta Soup": 5.0},
        "u4": {"Alpha Curry": 1.0, "Beta Curry": 1.0, "Delta Soup": 5.0},
    }.items():
        for dish, rating in vals.items():
            _rate(code, dish, rating, u)

    candidates = {c["dish_name"]: c for c in mr.get_candidates_for_group(code, n_candidates=20)}
    assert set(candidates) == set(names)  # everyone is returned when n is large

    def _has_cf_evidence(dish):
        return any("similar family members" in e for e in candidates[dish]["evidence"])

    # Correlated pair earns the CF evidence; the rated-but-unlike dish and the
    # never-rated dish must NOT — pre-fix every rated dish self-boosted to
    # cf_score 1 and would have shown the evidence.
    assert _has_cf_evidence("Alpha Curry")
    assert _has_cf_evidence("Beta Curry")
    assert not _has_cf_evidence("Delta Soup")
    assert not _has_cf_evidence("Epsilon Salad")
