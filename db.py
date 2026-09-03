import pandas as pd
import os
import random
import string
from typing import Optional
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import SessionLocal, engine, run_schema_migrations
from backend import models

# Keep the schema up to date (adds new columns to pre-existing tables). Safe/
# idempotent; a no-op when the table does not exist yet.
run_schema_migrations()

# ---------------------------------------------------------------------------
# RATING UPDATE POLICY (exponential moving average)
# ---------------------------------------------------------------------------
# Rating rows are unique on (group_code, dish_id, user_name), so re-rating the
# same dish must update the single stored row rather than append history. A
# plain overwrite lets one fresh submission erase all earlier evidence and lets
# serving frequency (which is itself driven by the score) refresh a dish's
# entire rating signal — an artificial positive-feedback loop.
#
# Instead, an update damps the stored value toward the new observation:
#
#     stored = BETA * fresh + (1 - BETA) * stored
#
# Repeated ratings therefore converge toward the member's repeated preference
# (each exposure contributes) instead of replacing it with the last draw.
# New ratings (no existing row) store the submitted value directly. The week /
# day / comment fields still track the latest submission exactly as before.
RATING_EMA_BETA = 0.5

# -----------------------------------
# COMPATIBILITY WRAPPERS
# -----------------------------------

# Canonical weekday order used wherever saved schedules are surfaced.
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def extract_group_code(filename, prefix, suffix=".csv"):
    code = filename[len(prefix):-len(suffix)]
    if "_week" in code:
        code = code.split("_week")[0]
    return code

def load_data(filename: str) -> pd.DataFrame:
    """
    Compatibility layer to intercept CSV file reads and query SQLAlchemy instead.
    """
    db = SessionLocal()
    try:
        if filename == "groups.csv":
            query = db.query(models.Group.group_code, models.Group.family_name, models.Group.creator)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("group_"):
            code = extract_group_code(filename, "group_")
            query = db.query(models.Member.name, models.Member.likes, models.Member.dislikes).filter(models.Member.group_code == code)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("dishes_"):
            code = extract_group_code(filename, "dishes_")
            # "dishes" = the rotation available to recommendations/scheduling.
            # Dishes still awaiting poll approval carry source="Pending" and are
            # excluded here until promoted to source="Poll".
            query = (
                db.query(models.Dish.name.label("dish"), models.Dish.source)
                .filter(models.Dish.group_code == code, models.Dish.source != "Pending")
            )
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("ratings_"):
            code = extract_group_code(filename, "ratings_")
            query = db.query(models.Dish.name.label("dish"), models.Rating.user_name.label("user"), models.Rating.rating, models.Rating.week, models.Rating.day, models.Rating.comment).join(models.Dish).filter(models.Rating.group_code == code)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("poll_votes_"):
            code = extract_group_code(filename, "poll_votes_")
            query = db.query(models.Dish.name.label("dish"), models.PollVote.user_name.label("user"), models.PollVote.vote).join(models.Dish).filter(models.PollVote.group_code == code)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("poll_"):
            code = extract_group_code(filename, "poll_")
            # aggregate votes
            query = db.query(models.Dish.name.label("dish"), func.sum(models.PollVote.vote).label("votes")).join(models.PollVote).filter(models.PollVote.group_code == code).group_by(models.Dish.name)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("pending_"):
            code = extract_group_code(filename, "pending_")
            query = db.query(models.PendingSuggestion.dish_name.label("dish"), models.PendingSuggestion.suggester).filter(models.PendingSuggestion.group_code == code)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("served_log_"):
            code = extract_group_code(filename, "served_log_")
            query = db.query(models.ServedLog.day, models.Dish.name.label("dish"), models.ServedLog.week).join(models.Dish).filter(models.ServedLog.group_code == code)
            return pd.read_sql(query.statement, db.bind)
            
        if filename.startswith("schedule_"):
            parts = filename.split("_week")
            code = parts[0].replace("schedule_", "").replace(".csv", "")
            if len(parts) > 1:
                week = int(parts[1].replace(".csv", ""))
                query = db.query(models.ScheduleEntry.day.label("Day"), models.Dish.name.label("Dish"), models.ScheduleEntry.reason.label("Reason")).join(models.Dish).filter(models.ScheduleEntry.group_code == code, models.ScheduleEntry.week == week)
            else:
                # current week? return max week
                max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == code).scalar()
                if not max_week: return pd.DataFrame()
                query = db.query(models.ScheduleEntry.day.label("Day"), models.Dish.name.label("Dish"), models.ScheduleEntry.reason.label("Reason")).join(models.Dish).filter(models.ScheduleEntry.group_code == code, models.ScheduleEntry.week == max_week)
            df_out = pd.read_sql(query.statement, db.bind)
            if not df_out.empty and "Day" in df_out.columns:
                order_map = {d: i for i, d in enumerate(WEEKDAY_ORDER)}
                df_out["__day_order"] = df_out["Day"].map(order_map).fillna(len(WEEKDAY_ORDER))
                df_out = df_out.sort_values("__day_order").drop(columns=["__day_order"]).reset_index(drop=True)
            return df_out
            
        return pd.DataFrame()
    finally:
        db.close()

def save_data(df: pd.DataFrame, filename: str):
    """
    Intercept CSV writes and convert them into SQLAlchemy inserts/updates.
    This handles saving schedules, served logs, and basic polls.
    """
    db = SessionLocal()
    try:
        if filename.startswith("schedule_"):
            parts = filename.split("_week")
            code = parts[0].replace("schedule_", "").replace(".csv", "")
            week = None
            if len(parts) > 1:
                week = int(parts[1].replace(".csv", ""))
            elif not df.empty and "week" in df.columns:
                week = int(df["week"].iloc[0])
            
            if week:
                # Delete existing schedule for this week and group to replace
                db.query(models.ScheduleEntry).filter(models.ScheduleEntry.group_code == code, models.ScheduleEntry.week == week).delete()
                
                # Fetch dish map
                dishes = {d.name.lower(): d.id for d in db.query(models.Dish).filter(models.Dish.group_code == code).all()}
                
                for _, row in df.iterrows():
                    dish_name = str(row.get('Dish', '')).strip()
                    day = str(row.get('Day', '')).strip()
                    reason = str(row.get('Reason', row.get('reason', '')) or '').strip() or None
                    if dish_name.lower() in dishes:
                        se = models.ScheduleEntry(group_code=code, dish_id=dishes[dish_name.lower()], week=week, day=day, scheduled_date=datetime.now(), reason=reason)
                        db.add(se)
                db.commit()

        elif filename.startswith("served_log_"):
            code = extract_group_code(filename, "served_log_")
            dishes = {d.name.lower(): d.id for d in db.query(models.Dish).filter(models.Dish.group_code == code).all()}
            
            for _, row in df.iterrows():
                dish_name = str(row.get('dish', '')).strip()
                day = str(row.get('day', '')).strip()
                week = row.get('week', 1)
                if pd.isna(week): week = 1
                if dish_name.lower() in dishes:
                    # Check existing
                    existing = db.query(models.ServedLog).filter(models.ServedLog.group_code == code, models.ServedLog.dish_id == dishes[dish_name.lower()], models.ServedLog.day == day, models.ServedLog.week == int(week)).first()
                    if not existing:
                        sl = models.ServedLog(group_code=code, dish_id=dishes[dish_name.lower()], day=day, week=int(week))
                        db.add(sl)
            db.commit()

        elif filename.startswith("ratings_"):
            # Ratings are persisted through the SQLAlchemy Rating model (the
            # same upsert semantics as db.rate_dish). Only the CSV-style
            # filename is used to identify the group.
            code = extract_group_code(filename, "ratings_")
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = [str(c).strip().lower() for c in df.columns]

                def _clean(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return str(v).strip()

                dish_map = {d.name.lower(): d.id for d in db.query(models.Dish).filter(models.Dish.group_code == code).all()}

                for _, row in df.iterrows():
                    dish_name = str(_clean(row.get("dish")) or "")
                    user = str(_clean(row.get("user", row.get("user_name"))) or "")
                    raw_rating = row.get("rating")
                    if not dish_name or dish_name.lower() not in dish_map or raw_rating is None or pd.isna(raw_rating):
                        continue
                    dish_id = dish_map[dish_name.lower()]
                    week_val = row.get("week")
                    week_int = int(week_val) if week_val is not None and not pd.isna(week_val) else None
                    day = _clean(row.get("day")) or None
                    comment = _clean(row.get("comment"))

                    existing = (
                        db.query(models.Rating)
                        .filter(models.Rating.group_code == code, models.Rating.dish_id == dish_id, models.Rating.user_name == user)
                        .first()
                    )
                    if existing:
                        # EMA damped update (see RATING_EMA_BETA) — identical
                        # semantics to db.rate_dish; week/day/comment still track
                        # the latest submission.
                        existing.rating = round(RATING_EMA_BETA * float(raw_rating) + (1.0 - RATING_EMA_BETA) * float(existing.rating), 2)
                        if week_int is not None:
                            existing.week = week_int
                        if day:
                            existing.day = day
                        existing.comment = comment
                    else:
                        db.add(models.Rating(
                            group_code=code,
                            dish_id=dish_id,
                            user_name=user,
                            rating=float(raw_rating),
                            week=week_int if week_int is not None else 1,
                            day=day,
                            comment=comment,
                        ))
                db.commit()
    finally:
        db.close()

# -----------------------------------
# FILE PATH HELPERS
# -----------------------------------
def get_group_file(group_code: str): return f"group_{group_code}.csv"
def get_poll_votes_file(group_code: str): return f"poll_votes_{group_code}.csv"
def get_pending_file(group_code: str): return f"pending_{group_code}.csv"
def get_dishes_file(group_code: str): return f"dishes_{group_code}.csv"
def get_rating_file(group_code: str): return f"ratings_{group_code}.csv"

def generate_group_code(length=6):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_code(length=6):
    return generate_group_code(length)

def create_family_group(family_name: str, creator_name: str) -> str:
    db = SessionLocal()
    try:
        while True:
            code = generate_group_code()
            if not group_exists(code):
                break
                
        new_group = models.Group(group_code=code, family_name=family_name, creator=creator_name)
        db.add(new_group)
        
        # default dishes list
        default_dishes = [
        # North Indian
        ("Paneer Butter Masala", "Default"), ("Chole Bhature", "Default"), ("Rajma Chawal", "Default"), 
        ("Dal Makhani", "Default"), ("Palak Paneer", "Default"), ("Kadai Paneer", "Default"), 
        ("Aloo Gobi", "Default"), ("Bhindi Masala", "Default"), ("Baingan Bharta", "Default"), 
        ("Malai Kofta", "Default"), ("Dum Aloo", "Default"), ("Shahi Paneer", "Default"), 
        ("Jeera Rice", "Default"), ("Vegetable Pulao", "Default"), ("Butter Naan", "Default"), 
        ("Tandoori Roti", "Default"), ("Laccha Paratha", "Default"), ("Matar Paneer", "Default"), 
        ("Methi Matar Malai", "Default"), ("Aloo Paratha", "Default"),
        # South Indian
        ("Masala Dosa", "Default"), ("Idli Sambar", "Default"), ("Medu Vada", "Default"), 
        ("Rava Dosa", "Default"), ("Uttapam", "Default"), ("Lemon Rice", "Default"), 
        ("Coconut Rice", "Default"), ("Tamarind Rice", "Default"), ("Curd Rice", "Default"), 
        ("Upma", "Default"), ("Pongal", "Default"), ("Appam", "Default"), 
        ("Puttu", "Default"), ("Rava Khichdi", "Default"),
        # Street Food / Chaat
        ("Pani Puri", "Default"), ("Sev Puri", "Default"), ("Dahi Puri", "Default"), 
        ("Pav Bhaji", "Default"), ("Bhel Puri", "Default"), ("Samosa Chaat", "Default"), 
        ("Aloo Tikki Chaat", "Default"), ("Ragda Patties", "Default"), ("Vada Pav", "Default"), 
        ("Misal Pav", "Default"), ("Dabeli", "Default"), ("Papdi Chaat", "Default"), 
        ("Chana Chaat", "Default"),
        # Italian / Fusion
        ("Veg Pizza", "Default"), ("Margherita Pizza", "Default"), ("Paneer Pizza", "Default"), 
        ("Pasta Arrabiata", "Default"), ("Alfredo Pasta", "Default"), ("Garlic Bread", "Default"), 
        ("Mac and Cheese", "Default"), ("Lasagna", "Default"), ("Bruschetta", "Default"), 
        ("Risotto", "Default"), ("Pesto Pasta", "Default"),
        # Indo-Chinese
        ("Veg Hakka Noodles", "Default"), ("Veg Fried Rice", "Default"), ("Gobi Manchurian", "Default"), 
        ("Chilli Paneer", "Default"), ("Spring Rolls", "Default"), ("Veg Momos", "Default"), 
        ("Schezwan Noodles", "Default"), ("Paneer Manchurian", "Default"), ("Hot and Sour Soup", "Default"), 
        ("Manchow Soup", "Default"),
        # Mexican / Western
        ("Veg Quesadilla", "Default"), ("Tacos", "Default"), ("Burritos", "Default"), 
        ("Nachos with Salsa", "Default"), ("Veg Burger", "Default"), ("French Fries", "Default"), 
        ("Veg Sandwich", "Default"), ("Grilled Cheese Sandwich", "Default"), ("Paneer Wrap", "Default"), 
        ("Falafel Wrap", "Default"), ("Hummus & Pita", "Default"),
        # Gujarati / Rajasthani
        ("Dhokla", "Default"), ("Khandvi", "Default"), ("Thepla", "Default"), 
        ("Undhiyu", "Default"), ("Dal Baati Churma", "Default"), ("Gatte Ki Sabzi", "Default"), 
        ("Kadhi Khichdi", "Default"),
        # Desserts / Sweets
        ("Gulab Jamun", "Default"), ("Rasgulla", "Default"), ("Jalebi", "Default"), 
        ("Kheer", "Default"), ("Gajar Halwa", "Default"), ("Shrikhand", "Default"), 
        ("Rasmalai", "Default"), ("Ice Cream", "Default"), ("Brownies", "Default"), 
        ("Chocolate Cake", "Default"),
        # Healthy / Light
        ("Vegetable Khichdi", "Default"), ("Oats Upma", "Default"), ("Moong Dal Cheela", "Default"), 
        ("Fruit Salad", "Default"), ("Sprouts Salad", "Default"), ("Tomato Soup", "Default"), 
        ("Vegetable Clear Soup", "Default")
    ]
        for dname, dsource in default_dishes:
            dish = models.Dish(group_code=code, name=dname, source=dsource)
            db.add(dish)
            
        member = models.Member(group_code=code, name=creator_name, likes="All vegetarian dishes", dislikes="None")
        db.add(member)
        db.commit()
        return code
    finally:
        db.close()

def save_group(group_name: str, head_name: str) -> str:
    return create_family_group(group_name, head_name)

def add_member(group_code: str, name: str, likes: str = "", dislikes: str = "") -> bool:
    db = SessionLocal()
    try:
        name_clean = str(name).strip()
        if not name_clean: return False
        
        member = db.query(models.Member).filter(models.Member.group_code == group_code, models.Member.name == name_clean).first()
        if member:
            if likes: member.likes = likes
            if dislikes: member.dislikes = dislikes
        else:
            member = models.Member(group_code=group_code, name=name_clean, likes=likes, dislikes=dislikes)
            db.add(member)
        db.commit()
        return True
    finally:
        db.close()

def group_exists(group_code) -> bool:
    db = SessionLocal()
    try:
        group = db.query(models.Group).filter(models.Group.group_code == str(group_code).strip().upper()).first()
        return group is not None
    finally:
        db.close()

def get_group_members_count(group_code: str) -> int:
    """
    Number of people in the family who can vote. Votes are cast by registered
    users (the app's roster); legacy Member rows are the fallback for groups
    that predate user accounts.
    """
    db = SessionLocal()
    try:
        users = db.query(models.User).filter(models.User.group_code == group_code).count()
        if users:
            return users
        return db.query(models.Member).filter(models.Member.group_code == group_code).count()
    finally:
        db.close()

# -----------------------------------
# DISH FUNCTIONS
# -----------------------------------
def add_dish(group_code: str, dish: str, source: str = "Poll"):
    db = SessionLocal()
    try:
        dish_clean = str(dish).strip()
        existing = db.query(models.Dish).filter(models.Dish.group_code == group_code, models.Dish.name == dish_clean).first()
        if not existing:
            new_dish = models.Dish(group_code=group_code, name=dish_clean, source=source)
            db.add(new_dish)
            db.commit()
    finally:
        db.close()

def get_dishes(group_code: str):
    """Rotation dishes only (approved dishes reach this list; pending do not)."""
    db = SessionLocal()
    try:
        return [
            d.name
            for d in db.query(models.Dish)
            .filter(models.Dish.group_code == group_code, models.Dish.source != "Pending")
            .all()
        ]
    finally:
        db.close()

# -----------------------------------
# PENDING SUGGESTIONS / POLLS
# -----------------------------------
def suggest_dish(group_code: str, dish: str, suggester: str) -> bool:
    db = SessionLocal()
    try:
        dish_clean = str(dish).strip()
        existing = db.query(models.PendingSuggestion).filter(models.PendingSuggestion.group_code == group_code, models.PendingSuggestion.dish_name == dish_clean).first()
        if not existing:
            ps = models.PendingSuggestion(group_code=group_code, dish_name=dish_clean, suggester=suggester)
            db.add(ps)
            db.commit()
            return True
        return False
    finally:
        db.close()

def get_pending_suggestions(group_code: str) -> pd.DataFrame:
    return load_data(get_pending_file(group_code))

def vote_dish(group_code: str, dish: str, user: str) -> bool:
    """
    Record a poll vote (one per user, re-voting overwrites).

    Votes are usually cast on pending suggestions that are not yet in the
    rotation and therefore have no Dish row yet. In that case a placeholder
    Dish row (source="Pending") is created so the vote can attach to it;
    pending rows stay invisible to recommendations/scheduling until the dish
    reaches majority and is promoted to source="Poll".

    Returns True when the vote was recorded. Returns False when the dish is
    unknown (not in the rotation and not a pending suggestion) — nothing is
    written.
    """
    db = SessionLocal()
    try:
        dish_clean = str(dish).strip()
        user_clean = str(user).strip()

        dish_obj = db.query(models.Dish).filter(
            models.Dish.group_code == group_code, models.Dish.name == dish_clean
        ).first()

        if not dish_obj:
            pending = db.query(models.PendingSuggestion).filter(
                models.PendingSuggestion.group_code == group_code,
                models.PendingSuggestion.dish_name == dish_clean,
            ).first()
            if not pending:
                return False  # unknown dish — nothing to vote on
            dish_obj = models.Dish(group_code=group_code, name=dish_clean, source="Pending")
            db.add(dish_obj)
            db.flush()

        existing = db.query(models.PollVote).filter(
            models.PollVote.group_code == group_code,
            models.PollVote.dish_id == dish_obj.id,
            models.PollVote.user_name == user_clean,
        ).first()
        if existing:
            existing.vote = 1
        else:
            v = models.PollVote(group_code=group_code, dish_id=dish_obj.id, user_name=user_clean, vote=1)
            db.add(v)
        db.commit()

        # Check majority (promotes the dish to the rotation when reached)
        _maybe_promote_pending_if_majority(group_code, dish_clean, db)
        return True
    finally:
        db.close()

def _update_poll_summary(group_code: str):
    pass # Managed dynamically via load_data

def get_poll_results(group_code: str) -> pd.DataFrame:
    return load_data(f"poll_{group_code}.csv")

def _maybe_promote_pending_if_majority(group_code: str, dish: str, db: Session = None) -> bool:
    close_db = False
    if not db:
        db = SessionLocal()
        close_db = True
    try:
        # Voters are the group's registered users; legacy Member rows are the
        # fallback for pre-account families.
        members = db.query(models.User).filter(models.User.group_code == group_code).count()
        if members == 0:
            members = db.query(models.Member).filter(models.Member.group_code == group_code).count()
        if members <= 0:
            return False

        dish_obj = db.query(models.Dish).filter(models.Dish.group_code == group_code, models.Dish.name == dish).first()
        if not dish_obj:
            return False

        votes = db.query(func.sum(models.PollVote.vote)).filter(models.PollVote.group_code == group_code, models.PollVote.dish_id == dish_obj.id).scalar() or 0

        if votes > (members / 2.0):
            # Promote to the rotation: Poll-sourced dishes are visible to the
            # recommendation engine. Also clear the suggestion and its votes.
            dish_obj.source = "Poll"
            db.query(models.PendingSuggestion).filter(models.PendingSuggestion.group_code == group_code, models.PendingSuggestion.dish_name == dish).delete()
            db.query(models.PollVote).filter(models.PollVote.group_code == group_code, models.PollVote.dish_id == dish_obj.id).delete()
            db.commit()
            return True
        return False
    finally:
        if close_db: db.close()

# -----------------------------------
# RATING FUNCTIONS
# -----------------------------------
def rate_dish(group_code: str, dish: str, rating: float, user_name: str = "", week: Optional[int] = None, overwrite: bool = True):
    db = SessionLocal()
    try:
        dish_clean = str(dish).strip()
        user_clean = str(user_name).strip()
        dish_obj = db.query(models.Dish).filter(models.Dish.group_code == group_code, models.Dish.name == dish_clean).first()
        if not dish_obj: return False
        
        fresh = float(rating)
        if overwrite:
            existing = db.query(models.Rating).filter(models.Rating.group_code == group_code, models.Rating.dish_id == dish_obj.id, models.Rating.user_name == user_clean).first()
            if existing:
                # EMA damped update (see RATING_EMA_BETA): converge toward the
                # member's repeated preference instead of replacing it.
                existing.rating = round(RATING_EMA_BETA * fresh + (1.0 - RATING_EMA_BETA) * float(existing.rating), 2)
                existing.week = int(week) if week else existing.week
            else:
                r = models.Rating(group_code=group_code, dish_id=dish_obj.id, user_name=user_clean, rating=fresh, week=int(week) if week else 1)
                db.add(r)
        else:
            r = models.Rating(group_code=group_code, dish_id=dish_obj.id, user_name=user_clean, rating=float(rating), week=int(week) if week else 1)
            db.add(r)
        db.commit()
        return True
    finally:
        db.close()

def get_average_ratings(group_code: str) -> pd.DataFrame:
    db = SessionLocal()
    try:
        query = db.query(models.Dish.name.label("dish"), func.avg(models.Rating.rating).label("average_rating")).join(models.Rating).filter(models.Rating.group_code == group_code).group_by(models.Dish.name)
        return pd.read_sql(query.statement, db.bind)
    finally:
        db.close()

def clear_poll_votes_for_group(group_code: str):
    db = SessionLocal()
    try:
        db.query(models.PollVote).filter(models.PollVote.group_code == group_code).delete()
        db.commit()
    finally:
        db.close()

# -----------------------------------
# CENTRALIZED WEEK MANAGEMENT
# -----------------------------------
# Week semantics (single source of truth, shared by the backend generate
# endpoint): the week a group is currently planning is the LATEST week that
# already has a saved plan (regenerating replaces that same week), or week 1
# when no plan exists yet (the first plan starts at the next logical week, 1).

def get_current_planning_week(group_code: str) -> int:
    db = SessionLocal()
    try:
        max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == group_code).scalar()
        return max_week or 1
    finally:
        db.close()

def get_current_active_week(group_code: str) -> int:
    db = SessionLocal()
    try:
        max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == group_code).scalar()
        return max_week or 1
    finally:
        db.close()

def get_plan_weeks(group_code: str):
    """Distinct weeks that have saved schedule rows for a group (ascending)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.ScheduleEntry.week)
            .filter(models.ScheduleEntry.group_code == group_code)
            .distinct()
            .order_by(models.ScheduleEntry.week)
            .all()
        )
        return [r[0] for r in rows]
    finally:
        db.close()

# -----------------------------------
# DATE HELPERS (Not heavily modified but needed)
# -----------------------------------
from datetime import datetime
