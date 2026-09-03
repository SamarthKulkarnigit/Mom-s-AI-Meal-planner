import os
import sys
import pandas as pd
from datetime import datetime

# Adjust path so we can import backend models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import SessionLocal, engine
from backend import models

# Ensure tables are created
models.Base.metadata.create_all(bind=engine)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_files_with_prefix(prefix):
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(".csv")]

def extract_group_code(filename, prefix, suffix=".csv"):
    code = filename[len(prefix):-len(suffix)]
    # for schedule, it might be {code}_week1
    if "_week" in code:
        code = code.split("_week")[0]
    return code

def main():
    db = SessionLocal()
    
    print("--- MIGRATION START ---")
    
    # 1. Groups (groups.csv)
    groups_file = os.path.join(DATA_DIR, "groups.csv")
    if os.path.exists(groups_file):
        try:
            df = pd.read_csv(groups_file)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        added, skipped = 0, 0
        for _, row in df.iterrows():
            code = str(row.get('group_code', '')).strip()
            name = str(row.get('family_name', '')).strip()
            creator = str(row.get('creator', '')).strip()
            if not code or code == 'nan': continue
            
            existing = db.query(models.Group).filter(models.Group.group_code == code).first()
            if not existing:
                group = models.Group(group_code=code, family_name=name, creator=creator)
                db.add(group)
                added += 1
            else:
                skipped += 1
        db.commit()
        print(f"Groups: inserted {added}, skipped {skipped}")

    # 2. Members (group_{code}.csv) -> Member table
    member_files = get_files_with_prefix("group_")
    member_files = [f for f in member_files if f != "groups.csv"]
    added_members, skipped_members = 0, 0
    for f in member_files:
        code = extract_group_code(f, "group_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            name = str(row.get('name', '')).strip()
            if not name or name == 'nan': continue
            likes = str(row.get('likes', ''))
            dislikes = str(row.get('dislikes', ''))
            
            existing = db.query(models.Member).filter(
                models.Member.group_code == code, models.Member.name == name
            ).first()
            if not existing:
                member = models.Member(group_code=code, name=name, likes=likes, dislikes=dislikes)
                db.add(member)
                added_members += 1
            else:
                skipped_members += 1
    db.commit()
    print(f"Members: inserted {added_members}, skipped {skipped_members}")

    # 3. Dishes (dishes_{code}.csv) -> Dish table
    dish_files = get_files_with_prefix("dishes_")
    added_dishes, skipped_dishes = 0, 0
    dish_map = {} # cache to find dish.id easily later
    for f in dish_files:
        code = extract_group_code(f, "dishes_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            name = str(row.get('dish', '')).strip()
            if not name or name == 'nan': continue
            source = str(row.get('source', 'Poll'))
            if source == 'nan': source = 'Poll'
            
            existing = db.query(models.Dish).filter(
                models.Dish.group_code == code, models.Dish.name == name
            ).first()
            if not existing:
                dish = models.Dish(group_code=code, name=name, source=source)
                db.add(dish)
                added_dishes += 1
            else:
                skipped_dishes += 1
    db.commit()
    print(f"Dishes: inserted {added_dishes}, skipped {skipped_dishes}")
    
    # Pre-load dish mapping
    for d in db.query(models.Dish).all():
        dish_map[(d.group_code, d.name.lower())] = d.id

    # 4. Ratings (ratings_{code}.csv) -> Rating table
    rating_files = get_files_with_prefix("ratings_")
    added_ratings, skipped_ratings = 0, 0
    for f in rating_files:
        code = extract_group_code(f, "ratings_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            dish_name = str(row.get('dish', '')).strip()
            user = str(row.get('user', '')).strip()
            rating_val = row.get('rating', 0.0)
            week = row.get('week', 1)
            if pd.isna(week): week = 1
            day = str(row.get('day', ''))
            comment = str(row.get('comment', ''))
            if comment == 'nan': comment = ''
            
            if not dish_name or dish_name == 'nan': continue
            
            dish_id = dish_map.get((code, dish_name.lower()))
            if not dish_id: continue # Can't map dish
            
            existing = db.query(models.Rating).filter(
                models.Rating.group_code == code,
                models.Rating.dish_id == dish_id,
                models.Rating.user_name == user
            ).first()
            
            if not existing:
                r = models.Rating(
                    group_code=code, dish_id=dish_id, user_name=user,
                    rating=float(rating_val), week=int(week), day=day, comment=comment
                )
                db.add(r)
                added_ratings += 1
            else:
                skipped_ratings += 1
    db.commit()
    print(f"Ratings: inserted {added_ratings}, skipped {skipped_ratings}")

    # 5. Poll Votes (poll_votes_{code}.csv) -> PollVote table
    poll_vote_files = get_files_with_prefix("poll_votes_")
    added_votes, skipped_votes = 0, 0
    for f in poll_vote_files:
        code = extract_group_code(f, "poll_votes_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            dish_name = str(row.get('dish', '')).strip()
            user = str(row.get('user', '')).strip()
            vote = row.get('vote', 1)
            
            if not dish_name or dish_name == 'nan': continue
            dish_id = dish_map.get((code, dish_name.lower()))
            if not dish_id: continue
            
            existing = db.query(models.PollVote).filter(
                models.PollVote.group_code == code,
                models.PollVote.dish_id == dish_id,
                models.PollVote.user_name == user
            ).first()
            if not existing:
                v = models.PollVote(group_code=code, dish_id=dish_id, user_name=user, vote=int(vote))
                db.add(v)
                added_votes += 1
            else:
                skipped_votes += 1
    db.commit()
    print(f"Poll Votes: inserted {added_votes}, skipped {skipped_votes}")

    # 6. Pending Suggestions (pending_{code}.csv) -> PendingSuggestion table
    pending_files = get_files_with_prefix("pending_")
    added_pending, skipped_pending = 0, 0
    for f in pending_files:
        code = extract_group_code(f, "pending_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            dish_name = str(row.get('dish', '')).strip()
            suggester = str(row.get('suggester', '')).strip()
            if not dish_name or dish_name == 'nan': continue
            
            existing = db.query(models.PendingSuggestion).filter(
                models.PendingSuggestion.group_code == code,
                models.PendingSuggestion.dish_name == dish_name
            ).first()
            if not existing:
                ps = models.PendingSuggestion(group_code=code, dish_name=dish_name, suggester=suggester)
                db.add(ps)
                added_pending += 1
            else:
                skipped_pending += 1
    db.commit()
    print(f"Pending Suggestions: inserted {added_pending}, skipped {skipped_pending}")

    # 7. Served Logs (served_log_{code}.csv) -> ServedLog table
    served_files = get_files_with_prefix("served_log_")
    added_served, skipped_served = 0, 0
    for f in served_files:
        code = extract_group_code(f, "served_log_")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            dish_name = str(row.get('dish', '')).strip()
            day = str(row.get('day', '')).strip()
            week = row.get('week', 1)
            if pd.isna(week): week = 1
            
            if not dish_name or dish_name == 'nan': continue
            dish_id = dish_map.get((code, dish_name.lower()))
            if not dish_id: continue
            
            existing = db.query(models.ServedLog).filter(
                models.ServedLog.group_code == code,
                models.ServedLog.dish_id == dish_id,
                models.ServedLog.day == day,
                models.ServedLog.week == int(week)
            ).first()
            if not existing:
                sl = models.ServedLog(group_code=code, dish_id=dish_id, day=day, week=int(week))
                db.add(sl)
                added_served += 1
            else:
                skipped_served += 1
    db.commit()
    print(f"Served Logs: inserted {added_served}, skipped {skipped_served}")

    # 8. Schedules (schedule_{code}_week*.csv) -> ScheduleEntry table
    schedule_files = [f for f in os.listdir(DATA_DIR) if f.startswith("schedule_") and f.endswith(".csv") and "_week" in f]
    added_sched, skipped_sched = 0, 0
    for f in schedule_files:
        parts = f.split("_week")
        code = parts[0].replace("schedule_", "")
        week_str = parts[1].replace(".csv", "")
        try:
            week = int(week_str)
        except ValueError:
            continue
            
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f))
        except pd.errors.EmptyDataError:
            continue
        for _, row in df.iterrows():
            day = str(row.get('Day', '')).strip()
            dish_name = str(row.get('Dish', '')).strip()
            if not dish_name or dish_name == 'nan': continue
            
            dish_id = dish_map.get((code, dish_name.lower()))
            if not dish_id: continue
            
            existing = db.query(models.ScheduleEntry).filter(
                models.ScheduleEntry.group_code == code,
                models.ScheduleEntry.week == week,
                models.ScheduleEntry.day == day
            ).first()
            if not existing:
                se = models.ScheduleEntry(group_code=code, dish_id=dish_id, scheduled_date=datetime.now(), week=week, day=day)
                db.add(se)
                added_sched += 1
            else:
                skipped_sched += 1
    db.commit()
    print(f"Schedule Entries: inserted {added_sched}, skipped {skipped_sched}")

    db.close()
    print("--- MIGRATION COMPLETE ---")

if __name__ == "__main__":
    main()
