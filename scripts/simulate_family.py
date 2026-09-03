#!/usr/bin/env python3
"""
scripts/simulate_family.py

Developer/demo simulation: run several weeks of a realistic simulated family
through the REAL recommendation system and the REAL persistence paths so the
existing Analytics / Ratings / Meal Scheduler pages can be inspected against a
rich, multi-week database.

Example:
    python scripts/simulate_family.py --weeks 6 --members 4 --seed 42
    python scripts/simulate_family.py --weeks 6 --db /tmp/my_demo.db

Important properties
--------------------
- Uses an ISOLATED SQLite database (fresh temp file under /tmp by default).
  The live development database (backend/data/mealplanner.db) is never
  touched unless --db points at it AND --yes is passed.
- DATABASE_URL is set BEFORE any project module is imported (engines bind to
  it at import time).
- The weekly plan comes from ml_recommender.generate_weekly_plan_for_group()
  — the exact deterministic recommender the production backend uses as its
  fallback. No new recommendation algorithm, no Gemini, no API keys.
- Plans are persisted at explicit week numbers through db.save_data(...)
  ("schedule_<code>_week<N>.csv"), the same DB-backed path the scheduler UI
  uses. Production week resolution (max week / replace-in-place) is untouched;
  the simulator simply writes rows for weeks 1..N.
- Served meals are mirrored into ServedLog through the same DB-backed path the
  scheduler UI uses.
- Ratings are generated from STABLE hidden preference profiles (per member,
  fixed for the whole run) plus small deterministic Gaussian noise and rare
  small "off day" dips, then persisted through the real ratings path
  (db.save_data("ratings_...")), preserving the production upsert semantics:
  Rating is unique on (group_code, dish_id, user_name), so a member re-rating
  the same dish in a later week UPDATES the row (and its week tag) rather than
  appending history.

The metrics reported each week are captured immediately after that week's
ratings are persisted (later weeks can move repeated member/dish rows to newer
week tags, so the DB is not a perfect immutable rating log — this is a
production model property, not something this script changes).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Make project packages importable no matter where the script is invoked from.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEV_DB_PATH = (PROJECT_ROOT / "backend" / "data" / "mealplanner.db").resolve()

DEFAULT_PASSWORD = "demo1234"
DEFAULT_WEEKS = 6
DEFAULT_SEED = 42

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --------------------------------------------------------------------------- #
# Hidden preference archetypes
# --------------------------------------------------------------------------- #
# The real rotation menu (~110 default dishes from db.create_family_group) is
# classified with the keyword rules below. Affinities are star values:
#   strong like 5.0 | like 4.5 | neutral 3.0-3.5 | dislike 2.0 | strong dislike 1.5
# Profiles are computed once from the dish name and stay fixed for the whole
# simulation. Small, deliberate overlap between members (crowd-pleasers, shared
# dishes such as Paneer Pizza / Dal Makhani) gives collaborative filtering a
# meaningful signal.

DESSERT_KW = [
    "gulab jamun", "rasgulla", "jalebi", "kheer", "shrikhand", "rasmalai",
    "halwa", "ice cream", "brownies", "chocolate cake", "mishti doi",
]
SOUTH_KW = [
    "dosa", "idli", "vada", "uttapam", "sambar", "curd rice", "lemon rice",
    "coconut rice", "tamarind rice", "upma", "pongal", "appam", "puttu",
]
LIGHT_HEALTHY_KW = [
    "khichdi", "oats", "cheela", "sprouts", "salad", "fruit", "soup",
    "dhokla", "khandvi", "thepla",
]
SPICY_KW = [
    "chole", "bhature", "chilli", "manchurian", "schezwan", "hot and sour",
    "vada pav", "misal", "ragda", "pav bhaji", "tandoori", "tikka", "kadai",
]
RICH_KW = [
    "butter masala", "shahi paneer", "malai kofta", "kofta",
    "chole bhature", "butter naan",
]
# Family crowd-pleasers: every member rates these >= ~4.2 unless the dish is in
# that member's explicit dislike category. This creates the small overlap that
# gives collaborative filtering signal.
CROWD_PLEASERS = {"Dal Makhani", "Masala Dosa", "Veg Pizza", "Vegetable Pulao"}

# North-Indian gravies / breads / rice that Member A likes but are not spicy.
A_MILD_LIKE_KW = ["paneer", "rajma", "chawal", "paratha", "naan", "roti", "pulao"]
A_DISLIKE_KW = DESSERT_KW + ["curd rice", "khichdi", "sprouts", "salad", "oats"]
# Member B's "very rich" dislikes are the RICH_KW list; everything else that is
# not south/light falls to neutral.
B_LIKE_EXTRA_KW = LIGHT_HEALTHY_KW + ["curd rice", "lemon rice", "coconut rice", "tamarind rice"]
C_MILD_KW = [
    "dal makhani", "pizza", "mac and cheese", "alfredo", "lasagna",
    "garlic bread", "bruschetta", "risotto", "dhokla", "khandvi",
    "thepla", "undhiyu",
]

# --------------------------------------------------------------------------- #
# Curated demo catalog (~47 dishes across 11 cuisine families)
# --------------------------------------------------------------------------- #
# Experiment: a purpose-built demo menu whose dish names are clearly separable
# by the member archetype rules below, unlike the production default menu
# (103 dishes) where many dishes are indistinguishable to the profiles. Used
# when --catalog curated (default). Dishes are inserted ONLY into the isolated
# simulation DB; nothing in production is touched.
CURATED_MENU = [
    # North Indian
    "Paneer Butter Masala", "Kadai Paneer", "Palak Paneer", "Chole Bhature",
    "Rajma Chawal", "Dal Makhani", "Aloo Gobi",
    # Punjabi
    "Paneer Tikka", "Shahi Paneer", "Butter Naan",
    # South Indian
    "Masala Dosa", "Idli Sambar", "Medu Vada", "Uttapam", "Upma",
    "Curd Rice", "Lemon Rice", "Coconut Rice",
    # Gujarati
    "Dhokla", "Khandvi", "Thepla",
    # Bengali
    "Luchi", "Cholar Dal", "Mishti Doi",
    # Indo-Chinese
    "Gobi Manchurian", "Chilli Paneer", "Veg Hakka Noodles",
    "Veg Fried Rice", "Veg Momos", "Schezwan Noodles",
    # Italian
    "Margherita Pizza", "Veg Pizza", "Alfredo Pasta", "Mac and Cheese",
    "Lasagna", "Garlic Bread",
    # Mexican
    "Veg Quesadilla", "Tacos", "Nachos with Salsa",
    # Light / simple
    "Vegetable Pulao", "Khichdi", "Moong Dal Cheela", "Tomato Soup",
    "Sprouts Salad",
    # Desserts
    "Gulab Jamun", "Rasgulla", "Jalebi", "Kheer", "Gajar Halwa",
    "Rasmalai",
]

MEMBER_ARCHETYPES = [
    {
        "username": "priya",
        "archetype": "North-Indian & spicy lover",
        "description": "likes paneer curries, chole bhature, pav bhaji, chilli paneer; dislikes mild/sweet dishes",
    },
    {
        "username": "arjun",
        "archetype": "South-Indian & light eater",
        "description": "likes dosa, idli sambar, curd rice; dislikes very rich dishes",
    },
    {
        "username": "meera",
        "archetype": "Mild & sweet tooth",
        "description": "likes desserts, dal makhani, pizza; dislikes very spicy dishes",
    },
    {
        "username": "vikram",
        "archetype": "Flexible eater",
        "description": "enjoys most dishes, mildly prefers family favorites",
    },
]


def _in(dish_name: str, keywords) -> bool:
    n = dish_name.lower()
    return any(k in n for k in keywords)


def affinity_a(dish: str) -> float:
    """Member A: North-Indian & spicy lover."""
    if dish in CROWD_PLEASERS:
        return 4.5
    if _in(dish, A_DISLIKE_KW):
        return 1.5
    if _in(dish, SPICY_KW):
        return 5.0
    if _in(dish, A_MILD_LIKE_KW):
        return 4.5
    return 3.5


def affinity_b(dish: str) -> float:
    """Member B: South-Indian & light eater."""
    if dish in CROWD_PLEASERS:
        return 4.5
    if _in(dish, RICH_KW):
        return 1.5
    if _in(dish, SOUTH_KW):
        return 5.0
    if _in(dish, B_LIKE_EXTRA_KW):
        return 4.0
    return 3.0


def affinity_c(dish: str) -> float:
    """Member C: mild & sweet tooth."""
    if dish in CROWD_PLEASERS:
        return 4.5
    if _in(dish, DESSERT_KW):
        return 5.0
    if _in(dish, C_MILD_KW) or _in(dish, ["paneer butter masala", "malai kofta", "shahi paneer"]):
        return 4.5
    if _in(dish, SPICY_KW):
        return 1.5
    return 3.5


def affinity_d(dish: str) -> float:
    """Member D: flexible eater — mildly positive everywhere, favors variety."""
    value = 3.8
    if dish in CROWD_PLEASERS:
        value += 0.5
    if _in(dish, SPICY_KW) or _in(dish, DESSERT_KW) or _in(dish, SOUTH_KW) or _in(dish, A_MILD_LIKE_KW):
        value += 0.3
    return min(4.5, round(value, 1))


AFFINITY_FN = {"priya": affinity_a, "arjun": affinity_b, "meera": affinity_c, "vikram": affinity_d}


def _round_half(value: float) -> float:
    return round(min(5.0, max(1.0, value)) * 2) / 2.0


def _comment_for(member: str, dish: str, rating: float) -> str:
    if rating >= 4.5:
        return "Loved it!"
    if rating <= 2.0:
        if member == "meera" and _in(dish, SPICY_KW):
            return "Too spicy for me."
        return "Not a favorite."
    if rating >= 3.5:
        return "Pretty good."
    return "Okay."


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Simulate several weeks of family behavior through the real recommender + DB persistence.",
    )
    p.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Number of weeks to simulate (default %(default)s).")
    p.add_argument("--members", type=int, default=4, choices=[3, 4],
                   help="Simulated members: 4 = priya/arjun/meera/vikram, 3 = drop vikram (default %(default)s).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default %(default)s).")
    p.add_argument("--group-name", default="Demo Family", help="Demo family name (default %(default)r).")
    p.add_argument("--password", default=DEFAULT_PASSWORD,
                   help="Demo password for every simulated user (default %(default)r). Documented demo credentials only.")
    p.add_argument("--catalog", choices=["curated", "standard"], default="curated",
                   help="Demo dish catalog: 'curated' = ~47 dishes spanning 11 cuisine families "
                        "(experiment default); 'standard' = the production default menu seeded by "
                        "db.create_family_group (~103 dishes, previous simulation behaviour). "
                        "Default %(default)s.")
    p.add_argument("--db", default=None, help="SQLite file to simulate into. Defaults to a fresh temp file under /tmp.")
    p.add_argument("--yes", action="store_true",
                   help="Allow --db to point at the live development database backend/data/mealplanner.db.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # ---- database isolation -------------------------------------------------
    if args.db:
        db_path = Path(args.db).expanduser().resolve()
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mealplanner_sim_"))
        db_path = tmp_dir / "simulation.db"

    if db_path == DEV_DB_PATH and not args.yes:
        sys.exit(
            "\nREFUSING to run: --db points at the live development database "
            f"{DEV_DB_PATH}.\n"
            "The simulator never modifies real data by default. Re-run with "
            "--db <other/path.db> for an isolated copy, or pass --yes if you "
            "really want to simulate into the development database.\n"
        )

    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    # Imports must happen AFTER DATABASE_URL is set (engines bind at import).
    import numpy as np
    import pandas as pd
    import bcrypt

    from backend.database import Base, engine, run_schema_migrations
    import backend.models  # noqa: F401  (registers models on Base)
    import db
    import ml_recommender

    Base.metadata.create_all(bind=engine)
    run_schema_migrations()

    from backend.database import SessionLocal
    from backend import models as M

    # ---- deterministic demo family -----------------------------------------
    seed = args.seed
    rng = np.random.RandomState(seed)

    selected = MEMBER_ARCHETYPES[: args.members]
    member_names = [m["username"] for m in selected]
    creator = member_names[0]

    if args.catalog == "standard":
        # Previous simulation behaviour: the production default menu (~103 dishes).
        group_code = db.create_family_group(args.group_name, creator)
    else:
        # Curated demo catalog (this experiment): ~47 dishes spanning 11 cuisine
        # families, inserted only into the isolated simulation DB.
        group_code = db.generate_group_code()
        s = SessionLocal()
        try:
            s.add(M.Group(group_code=group_code, family_name=args.group_name, creator=creator))
            s.add(M.Member(group_code=group_code, name=creator, likes="All vegetarian dishes", dislikes="None"))
            for dish_name in CURATED_MENU:
                s.add(M.Dish(group_code=group_code, name=dish_name, source="Default"))
            s.commit()
        finally:
            s.close()
    print(f"Created demo family: {args.group_name!r} group_code={group_code} catalog={args.catalog}")

    s = SessionLocal()
    try:
        for username in member_names:
            s.add(M.User(
                username=username,
                hashed_password=bcrypt.hashpw(args.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
                group_code=group_code,
            ))
        s.commit()
    finally:
        s.close()

    # Real rotation dishes (source != Pending), used for the alignment report.
    rotation = sorted(db.get_dishes(group_code))
    if len(rotation) < 7:
        sys.exit("Rotation menu has fewer than 7 dishes — cannot simulate.")

    # ---- helpers that mirror the production persistence paths ---------------
    def persist_plan(week, plan_df):
        df = plan_df.copy()
        df["week"] = week
        df["Reason"] = "Selected by the recommendation engine based on family preferences and variety."
        db.save_data(df, f"schedule_{group_code}_week{week}.csv")

    def mirror_served(week, plan_df):
        served_df = pd.DataFrame({
            "dish": plan_df["Dish"].tolist(),
            "day": plan_df["Day"].tolist(),
            "week": [week] * len(plan_df),
        })
        old = db.load_data(f"served_log_{group_code}.csv")
        if old is None or old.empty:
            combined = served_df.copy()
        else:
            combined = pd.concat([old, served_df], ignore_index=True)
        db.save_data(combined, f"served_log_{group_code}.csv")

    def persist_ratings(week, plan_df):
        """One rating per member per served meal, from hidden profile + noise."""
        rows = []
        for username in member_names:
            affinity = AFFINITY_FN[username]
            # Occasionally a member misses one meal that week (small realism).
            skip_day = None
            if rng.random() < 0.15:
                skip_day = int(rng.randint(0, len(plan_df)))
            for idx, (day, dish) in enumerate(zip(plan_df["Day"], plan_df["Dish"])):
                if idx == skip_day:
                    continue
                base = affinity(dish)
                off_day = -0.5 if rng.random() < 0.06 else 0.0
                noise = float(rng.normal(0.0, 0.25))
                rating = _round_half(base + noise + off_day)
                rows.append({
                    "dish": dish,
                    "user": username,
                    "rating": rating,
                    "week": week,
                    "day": day,
                    "comment": _comment_for(username, dish, rating),
                })
        if rows:
            db.save_data(pd.DataFrame(rows), f"ratings_{group_code}.csv")

    def db_week_snapshot(week):
        """Read back what actually landed in the DB for this week."""
        s = SessionLocal()
        try:
            ratings = (
                s.query(M.Rating.rating, M.Rating.user_name)
                .filter(M.Rating.group_code == group_code, M.Rating.week == week)
                .all()
            )
            served = (
                s.query(M.ServedLog)
                .filter(M.ServedLog.group_code == group_code, M.ServedLog.week == week)
                .count()
            )
            stored_plan = (
                s.query(M.ScheduleEntry)
                .filter(M.ScheduleEntry.group_code == group_code, M.ScheduleEntry.week == week)
                .count()
            )
            return ratings, served, stored_plan
        finally:
            s.close()

    # ---- weekly metrics accumulators ---------------------------------------
    weekly = []          # ordered dicts, one per week (computed at persist time)
    plan_sets = []       # dish-name sets per week, for repeat/overlap metrics

    print("\n" + "=" * 72)
    print("SIMULATING WEEKS")
    print("=" * 72)

    for week in range(1, args.weeks + 1):
        # 1. Real deterministic recommender (cold start on week 1).
        np.random.seed(seed + week)
        plan = ml_recommender.generate_weekly_plan_for_group(group_code)
        if plan is None or plan.empty or len(plan) < 7:
            sys.exit(f"Week {week}: recommender returned fewer than 7 dishes — cannot continue.")
        plan_df = plan[["Day", "Dish"]].head(7).reset_index(drop=True)
        plan_dishes = plan_df["Dish"].astype(str).tolist()

        # 2/3. Persist this week's plan + mirror served meals.
        persist_plan(week, plan_df)
        mirror_served(week, plan_df)

        # 4/5. Simulate consumption -> ratings, persisted through the real path.
        persist_ratings(week, plan_df)

        # Metrics — read back from the DB right now (see module docstring for
        # why they are captured here rather than recomputed at the end).
        rating_rows, served_count, stored_plan_count = db_week_snapshot(week)

        # Hidden-profile alignment (noiseless, defined against the served meals).
        member_alignment = {
            u: float(np.mean([AFFINITY_FN[u](d) for d in plan_dishes]))
            for u in member_names
        }
        overall_alignment = float(np.mean(list(member_alignment.values())))

        if rating_rows:
            values = [r[0] for r in rating_rows]
            avg_rating = float(np.mean(values))
            per_member = {}
            for u in member_names:
                uv = [r[0] for r in rating_rows if r[1] == u]
                per_member[u] = float(np.mean(uv)) if uv else float("nan")
            distribution = {}
            for v in sorted({round(x, 1) for x in values}):
                distribution[v] = sum(1 for x in values if round(x, 1) == v)
        else:
            avg_rating = float("nan")
            per_member = {u: float("nan") for u in member_names}
            distribution = {}

        # Repeat frequency: fraction of this week's dishes seen in ANY earlier week.
        earlier = set().union(*plan_sets) if plan_sets else set()
        repeat_count = sum(1 for d in plan_dishes if d in earlier)
        repeat_frac = repeat_count / len(plan_dishes) if plan_dishes else 0.0

        # Plan overlap with the immediately previous week (Jaccard).
        if plan_sets:
            prev = plan_sets[-1]
            cur = set(plan_dishes)
            union = prev | cur
            jaccard = len(prev & cur) / len(union) if union else 0.0
        else:
            jaccard = None

        weekly.append({
            "week": week,
            "plan": list(zip(DAYS, plan_dishes)),
            "avg_rating": avg_rating,
            "per_member": per_member,
            "n_ratings": len(rating_rows),
            "distribution": distribution,
            "alignment": overall_alignment,
            "member_alignment": member_alignment,
            "repeat_count": repeat_count,
            "repeat_frac": repeat_frac,
            "jaccard": jaccard,
            "served_rows": served_count,
            "stored_plan_rows": stored_plan_count,
        })
        plan_sets.append(set(plan_dishes))

    # ------------------------------------------------------------------ #
    # Console report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("SIMULATION SUMMARY")
    print("=" * 72)
    print(f"Family:      {args.group_name}")
    print(f"Group code:  {group_code}")
    print(f"Members:     {', '.join(member_names)} ({len(member_names)})")
    for m in selected:
        print(f"             - {m['username']}: {m['description']}")
    print(f"Weeks:       {args.weeks}")
    print(f"Seed:        {seed}")
    print(f"Dishes in rotation (catalog): {len(rotation)}")
    print(f"Database:    {db_path}")

    for w in weekly:
        print("\n" + "-" * 72)
        print(f"WEEK {w['week']}")
        print("-" * 72)
        print("Plan:")
        for day, dish in w["plan"]:
            print(f"  {day}: {dish}")
        print(f"Average rating (ratings tagged week {w['week']}): "
              f"{w['avg_rating']:.2f}  (n={w['n_ratings']})")
        pm = ", ".join(f"{u}={v:.2f}" for u, v in w["per_member"].items())
        print(f"Per-member average rating: {pm}")
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(w["distribution"].items()))
        print(f"Rating distribution: {dist or 'no ratings'}")
        am = ", ".join(f"{u}={v:.2f}" for u, v in w["member_alignment"].items())
        print(f"Hidden-profile alignment: {am}")
        print(f"Hidden-profile alignment (overall): {w['alignment']:.2f} (stars, noiseless)")
        print(f"Repeat meals from earlier weeks: {w['repeat_count']}/{len(w['plan'])} "
              f"({w['repeat_frac'] * 100:.0f}%)")
        if w["jaccard"] is None:
            print("Plan overlap vs previous week: n/a (first week)")
        else:
            print(f"Plan overlap vs previous week (Jaccard): {w['jaccard']:.2f}")

    print("\n" + "=" * 72)
    print("TREND")
    print("=" * 72)
    print(f"{'Week':>4} | {'Avg Rating':>10} | {'Alignment':>9} | {'Repeat %':>8} | {'Plan Overlap':>12}")
    print("-" * 60)
    for w in weekly:
        overlap = "n/a" if w["jaccard"] is None else f"{w['jaccard']:.2f}"
        avg = "n/a" if w["avg_rating"] != w["avg_rating"] else f"{w['avg_rating']:.2f}"
        print(f"{w['week']:>4} | {avg:>10} | {w['alignment']:>9.2f} | "
              f"{w['repeat_frac'] * 100:>7.0f}% | {overlap:>12}")

    # ---- per-member most/least aligned dishes (full rotation) ---------------
    print("\n" + "=" * 72)
    print("HIDDEN PROFILE: MOST / LEAST ALIGNED DISHES")
    print("=" * 72)
    for username in member_names:
        affinity = AFFINITY_FN[username]
        ranked = sorted(rotation, key=lambda d: (affinity(d), d), reverse=True)
        top = ", ".join(ranked[:5])
        bottom = ", ".join(ranked[-5:])
        print(f"\n{username} ({dict((m['username'], m['archetype']) for m in MEMBER_ARCHETYPES)[username]}):")
        print(f"  Most aligned : {top}")
        print(f"  Least aligned: {bottom}")

    # ---- explanation --------------------------------------------------------
    print("\n" + "=" * 72)
    print("WHAT THIS SIMULATION DOES (AND DOES NOT) SHOW")
    print("=" * 72)
    print("""
- Week 1 is a cold-start week: no ratings exist yet, so the recommender's
  rating/collaborative signals are zero and the plan is content/variety-driven.
- Ratings are generated from STABLE hidden preference profiles per member
  (fixed for the whole run), not from per-week random draws, and include small
  deterministic Gaussian noise plus rare small off-day dips.
- Each weekly plan is produced by the REAL deterministic recommender
  (ml_recommender.generate_weekly_plan_for_group), the same function the
  production backend uses as its fallback path.
- Gemini is intentionally NOT used: the simulation evaluates the real
  recommendation + persistence pipeline, not the LLM. In production, Gemini is
  an organizer of grounded recommender candidates, not the component evaluated
  here.
- Weekly metrics are captured immediately after each week's ratings are
  persisted because the production Rating model is unique on
  (group_code, dish_id, user_name): if a member rates the same dish again in a
  later week, the row is UPDATED (rating and week tag) rather than appended.
  The DB is therefore not a perfect immutable per-week rating log, and the
  existing Analytics weekly trend must not be read as one. This script does not
  change that production schema.
- 'Hidden-profile alignment' is the mean noiseless affinity (stars) of each
  member for the meals served that week — a measured signal of how well the
  served plan matches the simulated preferences. It is not 'accuracy' or a
  prediction score.
""")

    # ---- how to open this DB in the real app --------------------------------
    print("=" * 72)
    print("OPEN THIS DATABASE IN THE REAL APP")
    print("=" * 72)
    print(f"""
Database:    {db_path}
Group code:  {group_code}
Usernames:   {', '.join(member_names)}
Password:    {args.password}   (documented demo credentials only)

Launch the API and the UI against this same database (do NOT use the app's
default DB), then log in with any username above and browse the pages:

    export DATABASE_URL=sqlite:///{db_path}
    # Terminal 1 — FastAPI backend
    DATABASE_URL=$DATABASE_URL uvicorn backend.main:app --reload --port 8000

    # Terminal 2 — Streamlit UI
    DATABASE_URL=$DATABASE_URL streamlit run main.py

Then inspect:
  - Meal Scheduler  -> saved plans for weeks 1..N
  - Daily Feedback  -> week selector lists weeks 1..N
  - Rate Dishes / View Ratings
  - Analytics       -> ratings/served data from the simulated weeks
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
