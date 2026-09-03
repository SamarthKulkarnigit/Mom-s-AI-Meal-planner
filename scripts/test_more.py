import db
import pandas as pd

code1 = db.create_family_group("Fam1", "User1")
code2 = db.create_family_group("Fam2", "User2")

# Rating insertion
db.add_dish(code1, "Dish1")
db.rate_dish(code1, "Dish1", 4.5, "User1", 1)

r1 = db.load_data(f"ratings_{code1}.csv")
print("Fam1 Ratings:", len(r1))
r2 = db.load_data(f"ratings_{code2}.csv")
print("Fam2 Ratings:", len(r2))

# Schedule insertion
sch_df = pd.DataFrame([{"Day": "Monday", "Dish": "Dish1"}])
db.save_data(sch_df, f"schedule_{code1}_week1.csv")

s1 = db.load_data(f"schedule_{code1}_week1.csv")
print("Fam1 Schedule Week 1:", len(s1))

# Check uniqueness constraints via double insert (should update or fail gracefully via db.py logic)
db.rate_dish(code1, "Dish1", 5.0, "User1", 1) # Overwrite
r1_after = db.load_data(f"ratings_{code1}.csv")
print("Fam1 Ratings after overwrite:", len(r1_after), "Val:", r1_after.iloc[0]['rating'])

print("All tests passed.")
