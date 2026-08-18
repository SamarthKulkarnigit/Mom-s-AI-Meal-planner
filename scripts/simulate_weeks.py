# scripts/simulate_weeks.py
import ml_recommender, db, random, os
import pandas as pd
from pathlib import Path

GROUP = "TTQ5UF"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

USERS = ["mom","dad","sam","sanju","ajja","keeri","saku","rahul","anita","sammy"]

def save_plan(plan_df, week):
    fname = DATA_DIR / f"schedule_{GROUP}_week{week}.csv"
    plan_df.to_csv(fname, index=False)
    print("Saved:", fname)

def simulate(n_weeks=4, ratings_per_week=8):
    for w in range(1, n_weeks+1):
        print(f"\n=== Week {w} generation ===")
        plan = ml_recommender.generate_weekly_plan_for_group(GROUP)
        if plan is None or plan.empty:
            print("Generator returned no plan. Stopping.")
            break
        save_plan(plan, w)

        # simulate ratings AFTER week is done
        new_ratings = []
        for _ in range(ratings_per_week):
            user = random.choice(USERS)
            # pick a random dish from that week's plan
            dish = plan.sample(1)["Dish"].values[0]
            rating = round(random.uniform(2.5, 5.0), 1)
            # call db.rate_dish with week
            db.rate_dish(GROUP, dish, rating, user_name=user, week=w)
            new_ratings.append((dish, user, rating, w))

        print(f"Appended {len(new_ratings)} simulated ratings for week {w} (sample):", new_ratings[:3])

if __name__ == "__main__":
    simulate(n_weeks=6, ratings_per_week=8)