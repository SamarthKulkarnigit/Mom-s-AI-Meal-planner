import pandas as pd
import os
import random
import string
from typing import Optional
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import SessionLocal, engine
from backend import models

# -----------------------------------
# COMPATIBILITY WRAPPERS
# -----------------------------------

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
            query = db.query(models.Dish.name.label("dish"), models.Dish.source).filter(models.Dish.group_code == code)
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
                query = db.query(models.ScheduleEntry.day.label("Day"), models.Dish.name.label("Dish")).join(models.Dish).filter(models.ScheduleEntry.group_code == code, models.ScheduleEntry.week == week)
            else:
                # current week? return max week
                max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == code).scalar()
                if not max_week: return pd.DataFrame()
                query = db.query(models.ScheduleEntry.day.label("Day"), models.Dish.name.label("Dish")).join(models.Dish).filter(models.ScheduleEntry.group_code == code, models.ScheduleEntry.week == max_week)
            return pd.read_sql(query.statement, db.bind)
            
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
                    if dish_name.lower() in dishes:
                        se = models.ScheduleEntry(group_code=code, dish_id=dishes[dish_name.lower()], week=week, day=day, scheduled_date=datetime.now())
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
            ("Paneer Butter Masala", "Default"), ("Chole Bhature", "Default"), ("Rajma Chawal", "Default"), 
            # ... abbreviated for script size, just a few ...
            ("Dal Makhani", "Default"), ("Palak Paneer", "Default")
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
    db = SessionLocal()
    try:
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
    db = SessionLocal()
    try:
        return [d.name for d in db.query(models.Dish).filter(models.Dish.group_code == group_code).all()]
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
    db = SessionLocal()
    try:
        dish_clean = str(dish).strip()
        user_clean = str(user).strip()
        
        dish_obj = db.query(models.Dish).filter(models.Dish.group_code == group_code, models.Dish.name == dish_clean).first()
        if not dish_obj: return False
        
        existing = db.query(models.PollVote).filter(models.PollVote.group_code == group_code, models.PollVote.dish_id == dish_obj.id, models.PollVote.user_name == user_clean).first()
        if existing:
            existing.vote = 1
        else:
            v = models.PollVote(group_code=group_code, dish_id=dish_obj.id, user_name=user_clean, vote=1)
            db.add(v)
        db.commit()
        
        # Check majority
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
        members = db.query(models.Member).filter(models.Member.group_code == group_code).count()
        if members <= 0: return False
        
        dish_obj = db.query(models.Dish).filter(models.Dish.group_code == group_code, models.Dish.name == dish).first()
        if not dish_obj: return False
        
        votes = db.query(func.sum(models.PollVote.vote)).filter(models.PollVote.group_code == group_code, models.PollVote.dish_id == dish_obj.id).scalar() or 0
        
        if votes > (members / 2.0):
            # Promote
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
        
        if overwrite:
            existing = db.query(models.Rating).filter(models.Rating.group_code == group_code, models.Rating.dish_id == dish_obj.id, models.Rating.user_name == user_clean).first()
            if existing:
                existing.rating = float(rating)
                existing.week = int(week) if week else existing.week
            else:
                r = models.Rating(group_code=group_code, dish_id=dish_obj.id, user_name=user_clean, rating=float(rating), week=int(week) if week else 1)
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
def get_current_planning_week(group_code: str) -> int:
    db = SessionLocal()
    try:
        max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == group_code).scalar()
        return (max_week or 0) + 1
    finally:
        db.close()

def get_current_active_week(group_code: str) -> int:
    db = SessionLocal()
    try:
        max_week = db.query(func.max(models.ScheduleEntry.week)).filter(models.ScheduleEntry.group_code == group_code).scalar()
        return max_week or 1
    finally:
        db.close()

# -----------------------------------
# DATE HELPERS (Not heavily modified but needed)
# -----------------------------------
from datetime import datetime
