# db.py
import pandas as pd
import os
import random
import string
from typing import Optional
from pathlib import Path
import shutil

DATA_DIR = Path("data")
os.makedirs(DATA_DIR, exist_ok=True)

GROUPS_FILE = os.path.join(DATA_DIR, "groups.csv")

# -----------------------------------
# CENTRALIZED SAFE DATA LOAD/SAVE
# -----------------------------------

def load_data(filename: str) -> pd.DataFrame:
    """
    Safely load a CSV file from DATA_DIR.
    Normalizes column headers to lowercase and strips whitespaces.
    Handles EmptyDataError and missing files gracefully.
    """
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        # clean columns
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading {filename}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, filename: str):
    """
    Safely save a DataFrame as a CSV file to DATA_DIR.
    Normalizes columns to lowercase, removes duplicates, and makes a backup.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = DATA_DIR / filename
    try:
        # Create backup if file exists
        if path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy(path, backup_path)
        
        # Clean columns
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Save
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"ERROR saving {filename}: {e}")

# -----------------------------------
# FILE PATH HELPERS
# -----------------------------------

def get_group_file(group_code: str):
    return f"group_{group_code}.csv"

def get_poll_votes_file(group_code: str):
    return f"poll_votes_{group_code}.csv"

def get_pending_file(group_code: str):
    return f"pending_{group_code}.csv"

def get_dishes_file(group_code: str):
    return f"dishes_{group_code}.csv"

def get_rating_file(group_code: str):
    return f"ratings_{group_code}.csv"

# -----------------------------------
# FAMILY GROUP / MEMBERS HELPERS
# -----------------------------------

def generate_group_code(length=6):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def generate_code(length=6):
    return generate_group_code(length)

def create_family_group(family_name: str, creator_name: str) -> str:
    while True:
        code = generate_group_code()
        if not group_exists(code):
            break

    groups_df = load_data("groups.csv")
    if groups_df.empty:
        groups_df = pd.DataFrame(columns=["group_code", "family_name", "creator"])

    new_group = pd.DataFrame([{
        "group_code": code,
        "family_name": family_name,
        "creator": creator_name
    }])
    groups_df = pd.concat([groups_df, new_group], ignore_index=True)
    save_data(groups_df, "groups.csv")

    # default dishes list (100 popular vegetarian and family dishes)
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
    dishes_df = pd.DataFrame([{"dish": d[0], "source": d[1]} for d in default_dishes])

    # starter files
    starter_data = {
        f"ratings_{code}.csv": pd.DataFrame(columns=["dish", "user", "rating", "week", "day", "comment"]),
        f"poll_{code}.csv": pd.DataFrame(columns=["dish", "votes"]),
        f"dishes_{code}.csv": dishes_df,
        f"served_log_{code}.csv": pd.DataFrame(columns=["day", "dish", "week"]),
        f"group_{code}.csv": pd.DataFrame(columns=["name", "likes", "dislikes"]),
        f"pending_{code}.csv": pd.DataFrame(columns=["dish", "suggester"]),
        f"poll_votes_{code}.csv": pd.DataFrame(columns=["dish", "user", "vote"])
    }

    for filename, df in starter_data.items():
        save_data(df, filename)

    # Register creator as first member
    add_member(code, creator_name, likes="All vegetarian dishes", dislikes="None")

    return code

def save_group(group_name: str, head_name: str) -> str:
    """Unifies with create_family_group to prevent duplicate logic."""
    return create_family_group(group_name, head_name)

def add_member(group_code: str, name: str, likes: str = "", dislikes: str = "") -> bool:
    filename = get_group_file(group_code)
    df = load_data(filename)
    if df.empty:
        df = pd.DataFrame(columns=["name", "likes", "dislikes"])
    
    name_clean = str(name).strip()
    if not name_clean:
        return False

    # Check if user already exists
    if name_clean.lower() in df["name"].astype(str).str.strip().str.lower().tolist():
        # Update preferences
        mask = df["name"].astype(str).str.strip().str.lower() == name_clean.lower()
        if likes:
            df.loc[mask, "likes"] = likes
        if dislikes:
            df.loc[mask, "dislikes"] = dislikes
        save_data(df, filename)
        return True

    new_member = pd.DataFrame([{
        "name": name_clean,
        "likes": likes,
        "dislikes": dislikes
    }])
    df = pd.concat([df, new_member], ignore_index=True)
    save_data(df, filename)
    return True

def group_exists(group_code) -> bool:
    df = load_data("groups.csv")
    if df.empty or "group_code" not in df.columns:
        return False
    codes = df["group_code"].astype(str).str.strip().str.upper().tolist()
    return str(group_code).strip().upper() in codes

def get_group_members_count(group_code: str) -> int:
    df = load_data(get_group_file(group_code))
    return len(df)

# -----------------------------------
# DISH FUNCTIONS
# -----------------------------------

def add_dish(group_code: str, dish: str, source: str = "Poll"):
    filename = f"dishes_{group_code}.csv"
    df = load_data(filename)
    if df.empty:
        df = pd.DataFrame(columns=["dish", "source"])
    dish_clean = str(dish).strip()
    if dish_clean.lower() not in df["dish"].astype(str).str.strip().str.lower().tolist():
        new_row = pd.DataFrame([{"dish": dish_clean, "source": source}])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df, filename)

def get_dishes(group_code: str):
    df = load_data(f"dishes_{group_code}.csv")
    if not df.empty and "dish" in df.columns:
        return df["dish"].astype(str).tolist()
    return []

# -----------------------------------
# PENDING SUGGESTIONS / POLLS
# -----------------------------------

def suggest_dish(group_code: str, dish: str, suggester: str) -> bool:
    filename = get_pending_file(group_code)
    df = load_data(filename)
    if df.empty:
        df = pd.DataFrame(columns=["dish", "suggester"])
    dish_clean = str(dish).strip()
    if any(dish_clean.lower() == str(x).strip().lower() for x in df["dish"].astype(str).tolist()):
        return False
    new_row = pd.DataFrame([{"dish": dish_clean, "suggester": suggester}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df, filename)
    return True

def get_pending_suggestions(group_code: str) -> pd.DataFrame:
    return load_data(get_pending_file(group_code))

def vote_dish(group_code: str, dish: str, user: str) -> bool:
    filename = get_poll_votes_file(group_code)
    df = load_data(filename)
    if df.empty:
        df = pd.DataFrame(columns=["dish", "user", "vote"])

    dish_clean = str(dish).strip()
    user_clean = str(user).strip()

    # remove existing vote (overwrite)
    if not df.empty:
        mask = ~((df["dish"].astype(str).str.strip().str.lower() == dish_clean.lower()) &
                 (df["user"].astype(str).str.strip().str.lower() == user_clean.lower()))
        df = df[mask].copy()

    new_row = pd.DataFrame([{"dish": dish_clean, "user": user_clean, "vote": 1}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df, filename)

    _update_poll_summary(group_code)

    try:
        _maybe_promote_pending_if_majority(group_code, dish_clean)
    except Exception:
        pass
    return True

def _update_poll_summary(group_code: str):
    votes_df = load_data(get_poll_votes_file(group_code))
    if votes_df.empty:
        save_data(pd.DataFrame(columns=["dish", "votes"]), f"poll_{group_code}.csv")
        return
    summary = votes_df.groupby("dish", as_index=False)["vote"].sum().rename(columns={"vote": "votes"})
    save_data(summary, f"poll_{group_code}.csv")

def get_poll_results(group_code: str) -> pd.DataFrame:
    votes_df = load_data(get_poll_votes_file(group_code))
    if votes_df.empty:
        return pd.DataFrame(columns=["dish", "votes"])
    agg = votes_df.groupby("dish", as_index=False)["vote"].sum().rename(columns={"dish": "dish", "vote": "votes"})
    return agg

def _maybe_promote_pending_if_majority(group_code: str, dish: str) -> bool:
    members = get_group_members_count(group_code)
    if members <= 0:
        return False

    votes_df = load_data(get_poll_votes_file(group_code))
    if votes_df.empty:
        return False
    
    votes_df["dish"] = votes_df["dish"].astype(str).str.strip()
    count = int(votes_df[votes_df["dish"].str.lower() == dish.lower()]["vote"].sum())

    if count > (members / 2.0):
        # Promote to rotation
        add_dish(group_code, dish, source="Poll")
        
        # Remove from pending suggestions
        pending_file = get_pending_file(group_code)
        pending_df = load_data(pending_file)
        if not pending_df.empty:
            pending_df = pending_df[~(pending_df["dish"].astype(str).str.strip().str.lower() == dish.lower())]
            save_data(pending_df, pending_file)
            
        # Clean up votes for this dish
        votes_df = votes_df[~(votes_df["dish"].astype(str).str.strip().str.lower() == dish.lower())]
        save_data(votes_df, get_poll_votes_file(group_code))
        
        _update_poll_summary(group_code)
        return True
    return False

# -----------------------------------
# RATING FUNCTIONS
# -----------------------------------

def rate_dish(group_code: str, dish: str, rating: float, user_name: str = "", week: Optional[int] = None, overwrite: bool = True):
    filename = get_rating_file(group_code)
    df = load_data(filename)
    if df.empty:
        df = pd.DataFrame(columns=["dish", "user", "rating", "week", "day", "comment"])

    dish_clean = str(dish).strip()
    user_clean = str(user_name).strip()

    if overwrite and not df.empty:
        mask = ~((df["dish"].astype(str).str.strip().str.lower() == dish_clean.lower()) &
                 (df["user"].astype(str).str.strip().str.lower() == user_clean.lower()))
        df = df[mask].copy()

    new_row = {
        "dish": dish_clean,
        "user": user_clean,
        "rating": float(rating),
        "week": int(week) if week is not None else "",
        "day": "",
        "comment": ""
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(df, filename)
    return True

def get_average_ratings(group_code: str) -> pd.DataFrame:
    df = load_data(get_rating_file(group_code))
    if df.empty or "dish" not in df.columns or "rating" not in df.columns:
        return pd.DataFrame(columns=["dish", "average_rating"])

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    if df.empty:
        return pd.DataFrame(columns=["dish", "average_rating"])

    avg_df = df.groupby("dish")["rating"].mean().reset_index()
    avg_df.columns = ["dish", "average_rating"]
    return avg_df

def clear_poll_votes_for_group(group_code: str):
    save_data(pd.DataFrame(columns=["dish", "user", "vote"]), get_poll_votes_file(group_code))
    _update_poll_summary(group_code)

# -----------------------------------
# CENTRALIZED WEEK MANAGEMENT
# -----------------------------------

def get_current_planning_week(group_code: str) -> int:
    """Returns max(scheduled_weeks) + 1. Defaults to 1 if no schedules exist."""
    try:
        files = os.listdir(DATA_DIR)
    except Exception:
        return 1
    prefix = f"schedule_{group_code}_week"
    weeks = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".csv"):
            try:
                part = f[len(prefix):-4]
                weeks.append(int(part))
            except ValueError:
                pass
    return max(weeks) + 1 if weeks else 1

def get_current_active_week(group_code: str) -> int:
    """Returns max(scheduled_weeks). Defaults to 1 if no schedules exist."""
    try:
        files = os.listdir(DATA_DIR)
    except Exception:
        return 1
    prefix = f"schedule_{group_code}_week"
    weeks = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".csv"):
            try:
                part = f[len(prefix):-4]
                weeks.append(int(part))
            except ValueError:
                pass
    return max(weeks) if weeks else 1

# -----------------------------------
# AUTOMATIC MIGRATIONS
# -----------------------------------

def _migrate_groups_schema():
    groups_path = DATA_DIR / "groups.csv"
    if groups_path.exists():
        try:
            df = pd.read_csv(groups_path)
            if not df.empty:
                df.columns = [str(c).strip().lower() for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()]

                # Build cleaner schema while coalescing legacy column variants
                new_df = pd.DataFrame()

                # 1. group_code
                if "group_code" in df.columns and "code" in df.columns:
                    new_df["group_code"] = df["group_code"].fillna(df["code"])
                elif "group_code" in df.columns:
                    new_df["group_code"] = df["group_code"]
                elif "code" in df.columns:
                    new_df["group_code"] = df["code"]
                else:
                    new_df["group_code"] = pd.Series(dtype=str)

                # 2. family_name
                if "family_name" in df.columns and "groupname" in df.columns:
                    new_df["family_name"] = df["family_name"].fillna(df["groupname"])
                elif "family_name" in df.columns:
                    new_df["family_name"] = df["family_name"]
                elif "groupname" in df.columns:
                    new_df["family_name"] = df["groupname"]
                else:
                    new_df["family_name"] = pd.Series(dtype=str)

                # 3. creator
                if "creator" in df.columns and "headname" in df.columns:
                    new_df["creator"] = df["creator"].fillna(df["headname"])
                elif "creator" in df.columns:
                    new_df["creator"] = df["creator"]
                elif "headname" in df.columns:
                    new_df["creator"] = df["headname"]
                else:
                    new_df["creator"] = pd.Series(dtype=str)

                new_df = new_df.dropna(subset=["group_code"])
                new_df["group_code"] = new_df["group_code"].astype(str).str.strip().str.upper()
                new_df = new_df[new_df["group_code"] != ""]
                new_df = new_df.drop_duplicates(subset=["group_code"])

                new_df.to_csv(groups_path, index=False)
        except Exception as e:
            print(f"ERROR migrating groups schema: {e}")

# Run schema migration on import
_migrate_groups_schema()
