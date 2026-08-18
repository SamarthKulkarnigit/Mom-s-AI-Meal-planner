# scripts/seed_data.py
"""
Seed script to add vegetarian dishes + optional sample polls/ratings for a group.
Usage:
  # From project root, venv activated:
  python scripts/seed_data.py   # uses default GROUP below
  python scripts/seed_data.py --group TTQ5UF --with-ratings --with-polls
"""

import os
import pandas as pd
import argparse
import random
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_GROUP = "TTQ5UF"

DISHES = [
 "Aloo Gobi","Baingan Bharta","Bhindi Masala","Chole","Dal Tadka","Dal Makhani","Jeera Rice",
 "Vegetable Biryani","Paneer Butter Masala","Palak Paneer","Shahi Paneer","Kadai Paneer",
 "Mutter Paneer","Malai Kofta","Navratan Korma","Aloo Paratha","Plain Paratha","Roti","Naan","Puri",
 "Dosa","Masala Dosa","Idli","Upma","Poha","Vada Pav","Pav Bhaji","Pani Puri","Bhel Puri","Sev Puri",
 "Ragda Patties","Samosa","Kachori","Hakka Noodles","Vegetable Fried Rice","Spring Rolls",
 "Hara Bhara Kabab","Tandoori Broccoli","Corn Chaat","Aloo Tikki","Chole Bhature","Rajma","Khichdi",
 "Kathi Roll","Mix Veg Curry","Vegetable Cutlets","Paneer Tikka","Shakshuka","Noodles",
 "Malpua","Gulab Jamun","Jalebi","Ras Malai","Holige","Shahi Tukda","Pavlova","Dhokla","Handvo",
 "Thepla","Undhiyu","Methi Thepla","Masala Kulcha","Litti Chokha","Spinach Corn Salad","Greek Salad",
 "Caprese","Veg Lasagna","Veg Pizza","Falafel Wrap","Hummus & Pita","Paneer Bhurji"
]

SAMPLE_USERS = ["mom","dad","sam","sanju","ajja","keeri","saku","rahul","anita","sammy"]

def append_unique_dishes(group):
    file = DATA_DIR / f"dishes_{group}.csv"
    if file.exists():
        df = pd.read_csv(file)
        # unify column lower-case 'dish' to compare
        existing = set(df.iloc[:,0].astype(str).str.strip().str.lower().tolist())
    else:
        df = pd.DataFrame(columns=["dish","source"])
        existing = set()

    added = []
    for d in DISHES:
        if d.strip().lower() not in existing:
            df = pd.concat([df, pd.DataFrame([{"dish": d, "source": "Seed"}])], ignore_index=True)
            added.append(d)
            existing.add(d.strip().lower())

    df.to_csv(file, index=False)
    print(f"Appended {len(added)} dishes to {file}.")
    if added:
        print("Added (sample):", added[:10])
    return file, added

def append_sample_polls(group, percent_prob=0.4):
    file = DATA_DIR / f"poll_{group}.csv"
    if file.exists():
        df = pd.read_csv(file)
        existing = set(df['dish'].astype(str).str.strip().str.lower().tolist()) if 'dish' in df.columns else set()
    else:
        df = pd.DataFrame(columns=["dish","votes"])
        existing = set()

    # pick random subset of dishes to create poll entries for
    candidates = [d for d in DISHES if d.strip().lower() not in existing]
    random.shuffle(candidates)
    num = max(3, int(len(DISHES) * percent_prob))
    picks = candidates[:num]
    for d in picks:
        votes = random.randint(0, 10)
        df = pd.concat([df, pd.DataFrame([{"dish": d, "votes": votes}])], ignore_index=True)

    df.to_csv(file, index=False)
    print(f"Appended {len(picks)} poll rows to {file}.")
    return file

def append_sample_ratings(group, avg_ratings_per_dish=3):
    file = DATA_DIR / f"ratings_{group}.csv"
    if file.exists():
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["dish","user","rating"])

    # For each dish, add a few ratings (random users, random scores 1-5)
    for d in DISHES:
        n = random.randint(0, avg_ratings_per_dish)  # sometimes 0 ratings
        for _ in range(n):
            u = random.choice(SAMPLE_USERS)
            rating = round(random.uniform(2.5, 5.0), 1)  # bias toward positive so ML has signals
            df = pd.concat([df, pd.DataFrame([{"dish": d, "user": u, "rating": rating}])], ignore_index=True)

    df.to_csv(file, index=False)
    print(f"Appended ratings to {file}.")
    return file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", "-g", default=DEFAULT_GROUP, help="Group code to seed (e.g. TTQ5UF)")
    parser.add_argument("--with-polls", action="store_true", help="Also seed poll_<group>.csv")
    parser.add_argument("--with-ratings", action="store_true", help="Also seed ratings_<group>.csv")
    parser.add_argument("--ratings-per-dish", type=int, default=3, help="Average ratings per dish")
    args = parser.parse_args()

    group = args.group.strip()
    print("Seeding for group:", group)
    dishes_file, added = append_unique_dishes(group)

    if args.with_polls:
        append_sample_polls(group)

    if args.with_ratings:
        append_sample_ratings(group, avg_ratings_per_dish=args.ratings_per_dish)

    print("Done.")

if __name__ == "__main__":
    main()