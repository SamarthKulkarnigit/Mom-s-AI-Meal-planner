"""
tests/test_exploration.py

Focused tests for the bounded controlled-exploration slot added to the
deterministic recommendation planner (ml_recommender.py):

- A valid candidate ranked OUTSIDE the top-k pool can enter the final plan
- At most one exploration slot is ever introduced
- No eligible candidate -> fully personalized fallback (incl. cold start)
- Recently served dishes are not picked when older/never-served candidates exist
- The final plan keeps 7 unique, mutually non-similar dishes
- Existing fatigue / similarity behavior stays intact
- Selection is deterministic for a fixed seed/input
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402
import ml_recommender as mr  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --------------------------------------------------------------------------- #
# DB helpers
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
    """Insert ScheduleEntry history rows (the table the penalty/exploration read)."""
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
# Pure-function tests on _pick_exploration_dish (deterministic, no DB)
# --------------------------------------------------------------------------- #

# A full top-20 whose tail is genuinely weak (scores 0.90 .. 0.16), so an
# outside candidate with a modest but real family signal can clear the guard.
def _weak_tail_stats(outside_pairs):
    top = [("S%02d" % i, 0.90 - 0.02 * i) for i in range(14)]     # S00..S13: 0.90..0.64
    top += [("S14", 0.35), ("S15", 0.30), ("S16", 0.25),
            ("S17", 0.20), ("S18", 0.18), ("S19", 0.16)]
    rows = top + list(outside_pairs)
    df = pd.DataFrame(rows, columns=["dish", "final_score"])
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


# A top-20 that is uniformly strong (0.90 .. 0.52): no weak tail at all.
def _strong_stats(outside_pairs):
    top = [("S%02d" % i, 0.90 - 0.02 * i) for i in range(20)]     # S00..S19: 0.90..0.52
    rows = top + list(outside_pairs)
    df = pd.DataFrame(rows, columns=["dish", "final_score"])
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


def _pick(stats, chosen, fatigue=None, latest_served=None, cf_sim=None):
    return mr._pick_exploration_dish(
        stats, min(20, len(stats)), chosen, fatigue or {},
        latest_served if latest_served is not None else {"HistoryDish"},
        pd.DataFrame(),  # empty content sim -> pairwise rule trivially passes
        cf_sim if cf_sim is not None else pd.DataFrame(),
    )


def test_pick_returns_outside_top20_candidate_when_plan_has_weak_tail():
    """Requirement 1 (selector): a dish ranked below the top-k is chosen when
    the plan's weakest member is not strongly more relevant."""
    stats = _weak_tail_stats([("Novel Dish", 0.15), ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S18", "S19", "S04"]   # weakest = S19 (0.16)
    out = _pick(stats, chosen)
    assert out is not None
    cand, displaced = out
    assert cand == "Novel Dish"              # strongest eligible outside dish
    assert displaced == "S19"                # only the weakest member is displaced
    rank = list(stats["dish"]).index(cand) + 1
    assert rank > 20                          # genuinely outside the pool


def test_pick_never_displaces_highly_relevant_plan_member():
    """Guard: when every plan member is strongly relevant, a merely-new dish
    (whose ceiling is a small content-derived score) is rejected."""
    stats = _strong_stats([("Novel Dish", 0.15)])
    chosen = ["S00", "S01", "S02", "S03", "S04", "S05", "S06"]   # all >= ~0.78
    assert _pick(stats, chosen) is None


def test_pick_falls_back_when_no_plan_history():
    """Requirement 3 (cold start): no saved-plan week -> nothing is locked."""
    stats = _weak_tail_stats([("Novel Dish", 0.15), ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S18", "S19", "S04"]
    assert _pick(stats, chosen, latest_served=set()) is None


def test_pick_falls_back_when_all_outside_dishes_just_served():
    """Every candidate was served in the latest saved week -> no exploration."""
    stats = _strong_stats([("JustServed A", 0.50), ("JustServed B", 0.45)])
    chosen = ["S00", "S01", "S02", "S03", "S04", "S05", "S19"]
    assert _pick(stats, chosen,
                 latest_served={"JustServed A", "JustServed B"}) is None


def test_pick_falls_back_when_catalog_fits_top_k():
    """Fewer/equal dishes than top-k -> nothing exists outside the pool."""
    df = pd.DataFrame([(f"D{i}", 1.0 - 0.01 * i) for i in range(15)],
                      columns=["dish", "final_score"])
    chosen = [f"D{i}" for i in range(7)]
    assert _pick(df, chosen) is None


def test_recently_served_not_chosen_when_cleaner_candidate_exists():
    """Requirement 4: a just-served higher-scoring dish loses to an eligible
    never-served candidate."""
    stats = _strong_stats([("JustServed Star", 0.50), ("Older Gem", 0.45),
                           ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S04", "S05", "S19"]
    out = _pick(stats, chosen, latest_served={"JustServed Star"})
    assert out is not None
    assert out[0] == "Older Gem"


def test_fatigue_penalized_dish_not_chosen_when_clean_candidate_exists():
    """Candidates served within the fatigue horizon (factor < 1.0) are skipped
    while a never-served candidate is available."""
    stats = _strong_stats([("Recent Reheat", 0.50), ("Fresh Thing", 0.45),
                           ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S04", "S05", "S19"]
    out = _pick(stats, chosen, fatigue={"Recent Reheat": 0.75},
                latest_served={"SomethingElse"})
    assert out is not None
    assert out[0] == "Fresh Thing"


def test_pick_rejects_candidate_too_similar_to_remaining_plan():
    """The exploration dish must satisfy the >= 0.9 pairwise similarity rule
    against the six dishes that remain after the swap."""
    stats = _strong_stats([("Twin S00", 0.50), ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S04", "S05", "S19"]
    cf = pd.DataFrame(
        [[0.0 if a == b else (0.95 if {a, b} == {"Twin S00", "S00"} else 0.0)
          for b in stats["dish"]] for a in stats["dish"]],
        index=stats["dish"], columns=stats["dish"],
    )
    assert _pick(stats, chosen, cf_sim=cf) is None


def test_pick_is_deterministic():
    stats = _weak_tail_stats([("Novel Dish", 0.15), ("Bland Zeta", 0.00)])
    chosen = ["S00", "S01", "S02", "S03", "S18", "S19", "S04"]
    assert _pick(stats, chosen) == _pick(stats, chosen) == ("Novel Dish", "S19")


# --------------------------------------------------------------------------- #
# End-to-end planner tests (isolated SQLite DB)
# --------------------------------------------------------------------------- #

def _build_explorable_group():
    """Group with plan history, a rating-locked top-20, and one content-affine
    unrated dish ranked just OUTSIDE the pool (rank 21)."""
    code = _make_group("explore_e2e")
    names = [f"Curry{i}" for i in range(1, 21)]
    ids = _add_dishes(code, names + ["Curry1 Special", "Bland Zeta"])
    for i, name in enumerate(names, start=1):
        rating = 5.0 if i <= 10 else (4.0 if i <= 15 else 1.0)
        _rate(code, name, rating, "explore_u1", week=1)
    # plan history (latest saved week = 1) -> exploration is unlocked
    _add_schedule(code, {1: [ids[f"Curry{i}"] for i in range(1, 8)]})
    return code


def _plan_rank_counts(plan, stats):
    ranked = stats.sort_values("final_score", ascending=False).reset_index(drop=True)
    rank_of = {str(r["dish"]): i + 1 for i, r in ranked.iterrows()}
    return [rank_of[d] for d in plan["Dish"]]


def test_exploration_dish_enters_plan_when_outside_top20():
    """Requirement 1 (planner): with history present, some fixed seed yields a
    plan containing the rank-21 dish."""
    code = _build_explorable_group()
    fired = []
    for seed in range(1, 9):
        np.random.seed(seed)
        plan, stats = mr.generate_weekly_plan_for_group(code, return_stats=True)
        ranks = _plan_rank_counts(plan, stats)
        if max(ranks) > 20:
            fired.append((seed, plan, stats))
    assert fired, "no seed produced an exploration slot; scenario did not fire"
    seed, plan, stats = fired[0]
    ranks = _plan_rank_counts(plan, stats)
    outside = [d for d in plan["Dish"] if ranks[list(plan["Dish"]).index(d)] > 20]
    assert outside, "plan claims an outside dish but none found"
    assert outside == ["Curry1 Special"], f"unexpected outside dish: {outside}"


def test_at_most_one_exploration_slot_and_plan_stays_valid():
    """Requirements 2 & 5: every plan is a valid 7-day plan with at most one
    dish ranked outside the top-20 pool, all pairwise non-similar."""
    code = _build_explorable_group()
    for seed in range(1, 9):
        np.random.seed(seed)
        plan, stats = mr.generate_weekly_plan_for_group(code, return_stats=True)
        assert plan is not None and len(plan) == 7
        assert sorted(plan["Day"]) == sorted(DAYS)
        assert plan["Dish"].nunique() == 7
        ranks = _plan_rank_counts(plan, stats)
        assert sum(r > 20 for r in ranks) <= 1

        # The exploration dish must satisfy the >= 0.9 pairwise rule against
        # the six personalized dishes it sits next to (the personalized picks
        # themselves go through the main loop + padding, which is pre-existing
        # behavior covered by the fatigue tests).
        outside = [d for d, r in zip(plan["Dish"], ranks) if r > 20]
        if outside:
            dishes_df = db.load_data(f"dishes_{code}.csv")
            cs = mr._build_content_sim(dishes_df)
            ratings_df = db.load_data(f"ratings_{code}.csv")
            cfs = mr._build_cf_sim(ratings_df) if ratings_df is not None and not ratings_df.empty else pd.DataFrame()
            for cand in outside:
                for other in plan["Dish"]:
                    if other == cand:
                        continue
                    sim = 0.0
                    if not cfs.empty and cand in cfs.index and other in cfs.columns:
                        sim = max(sim, float(cfs.loc[cand, other]))
                    if not cs.empty and cand in cs.index and other in cs.columns:
                        sim = max(sim, float(cs.loc[cand, other]))
                    assert sim < 0.9, f"{cand} too similar to {other}"


def test_no_history_cold_start_falls_back_to_personalized():
    """Requirement 3 (planner): without a saved plan week every dish in the
    plan is inside the top-20 pool (fully personalized)."""
    code = _make_group("explore_coldstart")
    names = [f"Curry{i}" for i in range(1, 21)] + ["Curry1 Special", "Bland Zeta"]
    _add_dishes(code, names)
    for i, name in enumerate(names[:20], start=1):
        rating = 5.0 if i <= 10 else (4.0 if i <= 15 else 1.0)
        _rate(code, name, rating, "cold_u1", week=1)
    # NOTE: no _add_schedule -> no saved-plan history -> no exploration
    for seed in range(1, 5):
        np.random.seed(seed)
        plan, stats = mr.generate_weekly_plan_for_group(code, return_stats=True)
        assert max(_plan_rank_counts(plan, stats)) <= 20
        assert len(plan) == 7 and plan["Dish"].nunique() == 7


def test_deterministic_plans_for_fixed_seed():
    """Requirement 7: identical seed/input -> identical plan (exploration adds
    no randomness; the swap is deterministic)."""
    code = _build_explorable_group()
    np.random.seed(3)
    plan_a, _ = mr.generate_weekly_plan_for_group(code, return_stats=True)
    np.random.seed(3)
    plan_b, _ = mr.generate_weekly_plan_for_group(code, return_stats=True)
    assert list(plan_a["Dish"]) == list(plan_b["Dish"])
    assert list(plan_a["Day"]) == list(plan_b["Day"])
