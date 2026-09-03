"""
tests/test_fatigue_penalty.py

Focused tests for the soft recent-repetition / fatigue penalty added to the
deterministic recommendation planner (ml_recommender.py):

- No serving history      -> no penalty (cold start unchanged)
- Recently served dish    -> stronger penalty than an older one
- Older / beyond-horizon  -> smaller / zero penalty
- Highly preferred recent dish can still rank competitively (soft, capped)
- Recommendation output remains valid (7 unique days, dishes from rotation)
- The factor is deterministic and bounded
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402
import ml_recommender as mr  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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
    dish_ids = {}
    try:
        for name in names:
            d = m.Dish(group_code=code, name=name, source="Poll")
            s.add(d)
            s.flush()
            dish_ids[name] = d.id
        s.commit()
    finally:
        s.close()
    return dish_ids


def _add_schedule(code, dish_ids_by_week):
    """Insert ScheduleEntry history rows directly (the table the penalty reads)."""
    s = SessionLocal()
    try:
        for week, ids in dish_ids_by_week.items():
            for i, dish_id in enumerate(ids):
                day = DAYS[i % 7] if i < 7 else f"Slot{i}"
                s.add(m.ScheduleEntry(
                    group_code=code, dish_id=dish_id, week=week, day=day,
                    reason=None,
                ))
        s.commit()
    finally:
        s.close()


def _rate(code, dish_name, rating, user, week):
    assert db.rate_dish(code, dish_name, rating, user_name=user, week=week) is True


# --------------------------------------------------------------------------- #
# 1. No history -> no penalty (cold start unchanged)
# --------------------------------------------------------------------------- #

def test_no_history_means_no_penalty():
    code = _make_group("fatigue_no_history")
    names = [f"Dish {i}" for i in range(1, 16)]
    _add_dishes(code, names)

    assert mr._serving_fatigue_factor(code) == {}

    # Cold-start plan still works and is valid.
    np.random.seed(42)
    plan = mr.generate_weekly_plan_for_group(code)
    assert plan is not None and len(plan) == 7
    assert sorted(plan["Day"]) == sorted(DAYS)
    assert set(plan["Dish"]).issubset(set(names))


# --------------------------------------------------------------------------- #
# 2/3. Recency: recently served penalized more, older/horizon less/zero
# --------------------------------------------------------------------------- #

def test_recently_served_dish_penalized_more_than_older():
    code = _make_group("fatigue_recency")
    ids = _add_dishes(code, ["Recent A", "Recent B", "Older C", "Never D"])
    _add_schedule(code, {
        5: [ids["Recent A"], ids["Recent B"]],  # most recent saved week -> age 0
        2: [ids["Older C"]],                     # 3 weeks older -> age 3
        # "Never D" has no schedule row at all
    })

    factors = mr._serving_fatigue_factor(code)
    assert factors["Recent A"] == factors["Recent B"] == 1.0 - mr.FATIGUE_PENALTY_MAX
    # Older dish is penalized too, but less than the most recent one.
    assert factors["Older C"] < 1.0
    assert factors["Recent A"] < factors["Older C"]
    # Never-served dish: no factor entry -> caller treats as 1.0 (no penalty).
    assert "Never D" not in factors
    # Bounds: factor strictly inside (0.75, 1.0] for any penalized dish.
    assert all(1.0 - mr.FATIGUE_PENALTY_MAX <= f <= 1.0 for f in factors.values())


def test_penalty_decays_with_age_and_hits_zero_after_horizon():
    code = _make_group("fatigue_decay")
    ids = _add_dishes(code, ["Newest", "Mid", "Ancient"])
    horizon = mr.FATIGUE_PENALTY_HORIZON_WEEKS
    _add_schedule(code, {
        10: [ids["Newest"]],                     # age 0
        10 - 2: [ids["Mid"]],                    # age 2
        10 - horizon - 1: [ids["Ancient"]],      # age > horizon -> zero penalty
    })

    factors = mr._serving_fatigue_factor(code)
    assert factors["Newest"] < factors["Mid"] < 1.0
    # Beyond the horizon the dish is absent from the factor map, which the
    # caller treats exactly as factor 1.0 (no penalty).
    assert "Ancient" not in factors

    # Deterministic: repeated calls return identical results.
    assert factors == mr._serving_fatigue_factor(code)


# --------------------------------------------------------------------------- #
# 4. Soft cap: a highly preferred recent dish can still rank competitively
# --------------------------------------------------------------------------- #

def test_highly_preferred_recent_dish_still_ranks():
    code = _make_group("fatigue_preferred")
    names = [f"Filler {i}" for i in range(1, 13)] + ["Favourite Curry"]
    ids = _add_dishes(code, names)

    # The favourite is strongly liked by two members...
    for user in ("fatigue_u1", "fatigue_u2"):
        _rate(code, "Favourite Curry", 5.0, user, week=1)
        for i in range(1, 13):
            _rate(code, f"Filler {i}", 2.0, user, week=1)
    # ...yet it is also the dish served in the most recent saved week (age 0).
    _add_schedule(code, {2: [ids["Favourite Curry"]]})

    # Despite the max fatigue cut (age-0 factor 0.75), the favourite's adjusted
    # score must still be the top-ranked candidate (deterministic, seed-free).
    np.random.seed(42)
    _, stats = mr.generate_weekly_plan_for_group(code, return_stats=True)
    ranked = stats.sort_values("final_score", ascending=False).reset_index(drop=True)
    assert ranked.iloc[0]["dish"] == "Favourite Curry"
    assert ranked.iloc[0]["fatigue_factor"] == 1.0 - mr.FATIGUE_PENALTY_MAX  # cut applied

    # ...and the weighted A-Res sampler still surfaces it in practice (checked
    # across several fixed seeds so the assertion is not one lucky draw).
    seen_in_any_plan = False
    for seed in range(1, 6):
        np.random.seed(seed)
        plan = mr.generate_weekly_plan_for_group(code)
        assert plan is not None and len(plan) == 7
        if "Favourite Curry" in set(plan["Dish"]):
            seen_in_any_plan = True
        # Recommendation output validity with history present: 7 unique weekdays.
        assert sorted(plan["Day"]) == sorted(DAYS)
        assert set(plan["Dish"]).issubset(set(names))
    assert seen_in_any_plan


def test_plan_remains_valid_when_history_present():
    """Plans produced with a full serving history are still valid output."""
    code = _make_group("fatigue_valid")
    names = [f"Valid Dish {i}" for i in range(1, 12)]
    ids = _add_dishes(code, names)
    _add_schedule(code, {1: list(ids.values())})  # everything served week 1

    for user in ("fatigue_v1", "fatigue_v2"):
        for i, name in enumerate(names):
            _rate(code, name, 3.0 + (i % 3), user, week=1)

    np.random.seed(7)
    plan = mr.generate_weekly_plan_for_group(code)
    assert plan is not None and len(plan) == 7
    assert sorted(plan["Day"]) == sorted(DAYS)
    assert plan["Dish"].nunique() == 7
    assert set(plan["Dish"]).issubset(set(names))
