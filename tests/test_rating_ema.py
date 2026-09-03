"""
tests/test_rating_ema.py

Focused tests for the write-side exponential moving average applied to repeated
member-dish ratings (db.py):

    stored = RATING_EMA_BETA * fresh + (1 - RATING_EMA_BETA) * stored

- A new rating stores the exact submitted value (no EMA on first write)
- Updating an existing rating applies the EMA
- Repeated ratings converge toward the member's repeated preference instead of
  replacing the old value (no artificial single-draw refresh)
- Week / day / comment updates still track the latest submission
- Both write paths (db.rate_dish and db.save_data ratings branch) behave the
  same
- Stored values always stay within 1.0-5.0
- Unrelated member / dish / group ratings stay isolated
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd  # noqa: E402

import db  # noqa: E402
from backend.database import SessionLocal  # noqa: E402
from backend import models as m  # noqa: E402

BETA = db.RATING_EMA_BETA
assert BETA == 0.5


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


def _add_dish(code, name):
    s = SessionLocal()
    try:
        d = m.Dish(group_code=code, name=name, source="Poll")
        s.add(d)
        s.commit()
        return d.id
    finally:
        s.close()


def _stored_rating(code, dish_name, user_name):
    df = db.load_data(f"ratings_{code}.csv")
    row = df[
        (df["dish"].astype(str).str.strip() == dish_name)
        & (df["user"].astype(str).str.strip() == user_name)
    ]
    assert len(row) == 1, "expected exactly one stored row"
    return float(row.iloc[0]["rating"])


def _ema(*fresh_values):
    """Simulate the expected damped value after successive EMA updates."""
    stored = None
    for v in fresh_values:
        stored = v if stored is None else round(BETA * v + (1.0 - BETA) * stored, 2)
    return stored


# --------------------------------------------------------------------------- #
# db.rate_dish path
# --------------------------------------------------------------------------- #

def test_rate_dish_new_rating_stores_exact_value():
    code = _make_group("ema_new_rate")
    _add_dish(code, "Butter Paneer")
    assert db.rate_dish(code, "Butter Paneer", 4.0, user_name="u_ema", week=1) is True
    assert _stored_rating(code, "Butter Paneer", "u_ema") == 4.0


def test_rate_dish_update_applies_ema():
    code = _make_group("ema_update_rate")
    _add_dish(code, "Dal Tadka")
    db.rate_dish(code, "Dal Tadka", 4.0, user_name="u_ema", week=1)
    db.rate_dish(code, "Dal Tadka", 5.0, user_name="u_ema", week=2)
    # 0.5*5.0 + 0.5*4.0 = 4.5  (not the plain overwrite 5.0)
    assert _stored_rating(code, "Dal Tadka", "u_ema") == 4.5


def test_rate_dish_repeated_ratings_converge_not_replace():
    """Repeated strong (and weak) feedback converges toward the repeated
    preference rather than flipping the stored value on every single draw."""
    code = _make_group("ema_converge_rate")
    _add_dish(code, "Khichdi")
    # A member who genuinely likes the dish rates 5.0 several times after a 2.0.
    draws = [2.0, 5.0, 5.0, 5.0, 5.0]
    for week, v in enumerate(draws, start=1):
        db.rate_dish(code, "Khichdi", v, user_name="u_ema", week=week)
    expected = _ema(*draws)  # 2.0 -> 3.5 -> 4.25 -> 4.625 -> 4.8125 (rounded 4.81)
    stored = _stored_rating(code, "Khichdi", "u_ema")
    assert stored == expected
    # ...and it is NOT the last raw draw (5.0), so a single noisy 5.0 cannot
    # erase a long history, and the value stays inside [1, 5].
    assert stored < 5.0
    assert 1.0 <= stored <= 5.0


def test_rate_dish_week_tag_still_moves_on_update():
    code = _make_group("ema_week_rate")
    _add_dish(code, "Veg Pulao")
    db.rate_dish(code, "Veg Pulao", 3.0, user_name="u_ema", week=1)
    db.rate_dish(code, "Veg Pulao", 4.0, user_name="u_ema", week=7)
    df = db.load_data(f"ratings_{code}.csv")
    assert len(df) == 1
    assert int(df.iloc[0]["week"]) == 7          # tag moved to the supplied week
    assert float(df.iloc[0]["rating"]) == 3.5    # value damped


# --------------------------------------------------------------------------- #
# db.save_data ratings branch (Daily Feedback / simulation path)
# --------------------------------------------------------------------------- #

def test_save_data_new_rating_stores_exact_value():
    code = _make_group("ema_new_save")
    _add_dish(code, "Chana Masala")
    db.save_data(pd.DataFrame([{"dish": "Chana Masala", "user": "u_ema", "rating": 4.5,
                                "week": 1, "day": "Monday", "comment": "Good"}]), f"ratings_{code}.csv")
    assert _stored_rating(code, "Chana Masala", "u_ema") == 4.5


def test_save_data_update_applies_ema_and_tracks_week_day_comment():
    code = _make_group("ema_update_save")
    _add_dish(code, "Rajma Chawal")
    db.save_data(pd.DataFrame([{"dish": "Rajma Chawal", "user": "u_ema", "rating": 4.0,
                                "week": 1, "day": "Monday", "comment": "Nice"}]), f"ratings_{code}.csv")
    db.save_data(pd.DataFrame([{"dish": "Rajma Chawal", "user": "u_ema", "rating": 3.0,
                                "week": 2, "day": "Tuesday", "comment": "Less today"}]), f"ratings_{code}.csv")
    df = db.load_data(f"ratings_{code}.csv")
    assert len(df) == 1
    assert float(df.iloc[0]["rating"]) == 3.5     # 0.5*3.0 + 0.5*4.0
    assert int(df.iloc[0]["week"]) == 2           # week moved
    assert df.iloc[0]["day"] == "Tuesday"         # day tracked
    assert df.iloc[0]["comment"] == "Less today"  # comment replaced


def test_both_write_paths_share_ema_semantics():
    """The same sequence through db.rate_dish and through db.save_data yields
    the same damped stored value."""
    code = _make_group("ema_both_paths")
    _add_dish(code, "Idli Sambar")
    db.rate_dish(code, "Idli Sambar", 2.0, user_name="u_a", week=1)
    db.rate_dish(code, "Idli Sambar", 5.0, user_name="u_a", week=2)
    db.save_data(pd.DataFrame([{"dish": "Idli Sambar", "user": "u_b", "rating": 2.0,
                                "week": 1}]), f"ratings_{code}.csv")
    db.save_data(pd.DataFrame([{"dish": "Idli Sambar", "user": "u_b", "rating": 5.0,
                                "week": 2}]), f"ratings_{code}.csv")
    assert _stored_rating(code, "Idli Sambar", "u_a") == 3.5
    assert _stored_rating(code, "Idli Sambar", "u_b") == 3.5


def test_values_stay_in_range_after_many_updates():
    code = _make_group("ema_bounds")
    _add_dish(code, "Misal Pav")
    # alternate extreme feedback; EMA must stay inside [1, 5] the whole time
    values = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0]
    for week, v in enumerate(values, start=1):
        db.rate_dish(code, "Misal Pav", v, user_name="u_ema", week=week)
    df = db.load_data(f"ratings_{code}.csv")
    stored = float(df.iloc[0]["rating"])
    assert 1.0 <= stored <= 5.0
    assert stored == _ema(*values)


def test_updates_are_isolated_per_member_dish_and_group():
    """EMA is applied per (group, dish, user) row only; no cross-talk."""
    code = _make_group("ema_iso1")
    code2 = _make_group("ema_iso2")
    _add_dish(code, "Aloo Gobi")
    _add_dish(code, "Paneer Tikka")
    _add_dish(code2, "Aloo Gobi")

    # member u1 rates Aloo Gobi twice (EMA); u2 rates it once (raw 5.0)
    db.rate_dish(code, "Aloo Gobi", 2.0, user_name="u1", week=1)
    db.rate_dish(code, "Aloo Gobi", 4.0, user_name="u1", week=2)
    db.rate_dish(code, "Aloo Gobi", 5.0, user_name="u2", week=2)
    # u1's Paneer Tikka row (different dish) must stay at its raw first value
    db.rate_dish(code, "Paneer Tikka", 3.0, user_name="u1", week=1)
    # same dish in a different group, first rating raw
    db.rate_dish(code2, "Aloo Gobi", 1.0, user_name="u1", week=1)

    assert _stored_rating(code, "Aloo Gobi", "u1") == 3.0      # EMA of 2.0, 4.0
    assert _stored_rating(code, "Aloo Gobi", "u2") == 5.0      # single rating, raw
    assert _stored_rating(code, "Paneer Tikka", "u1") == 3.0   # untouched by others
    assert _stored_rating(code2, "Aloo Gobi", "u1") == 1.0     # group-isolated


def test_week_none_on_update_keeps_old_tag_but_still_emas():
    """rate_dish without a week keeps the existing week (unchanged behaviour)
    while the value is still damped."""
    code = _make_group("ema_week_none")
    _add_dish(code, "Dosa")
    db.rate_dish(code, "Dosa", 4.0, user_name="u_ema", week=3)
    db.rate_dish(code, "Dosa", 2.0, user_name="u_ema", week=None)
    df = db.load_data(f"ratings_{code}.csv")
    assert len(df) == 1
    assert int(df.iloc[0]["week"]) == 3
    assert float(df.iloc[0]["rating"]) == 3.0
