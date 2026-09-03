import db
import pandas as pd
from backend.database import SessionLocal
from backend.models import User, Group, Dish, Rating, PollVote

def main():
    print("--- VERIFICATION START ---")
    
    # Check if a group exists
    groups_df = db.load_data("groups.csv")
    print(f"Loaded {len(groups_df)} groups via db.load_data")
    
    # Try creating a new group via db
    code = db.create_family_group("Test Family", "Test User")
    print(f"Created group with code: {code}")
    
    # Add a member
    db.add_member(code, "Wife", likes="Pasta", dislikes="None")
    print(f"Added member 'Wife' to group {code}")
    
    # Add a dish
    db.add_dish(code, "Test Pizza", "Poll")
    print("Added dish 'Test Pizza'")
    
    # Add a rating
    db.rate_dish(code, "Test Pizza", 5.0, user_name="Wife", week=1)
    print("Rated 'Test Pizza'")
    
    # Verify ratings can be loaded
    ratings_df = db.load_data(f"ratings_{code}.csv")
    print(f"Loaded {len(ratings_df)} ratings for group {code}")
    if not ratings_df.empty:
        print(ratings_df.head())
        
    print("--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    main()
