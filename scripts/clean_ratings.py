import pandas as pd
import os

GROUP = "TTQ5UF"   # change group code if needed
PATH = f"data/ratings_{GROUP}.csv"

if not os.path.exists(PATH):
    print("Ratings file not found:", PATH)
else:
    df = pd.read_csv(PATH, dtype=str)

    print("Original head:")
    print(df.head(10))

    # helper to pick lowercase value first then Titlecase
    def coalesce_row(row, lower, title):
        v = None
        if lower in row and pd.notna(row[lower]) and str(row[lower]).strip() != "":
            v = row[lower]
        if (v is None or str(v).strip() == "") and title in row and pd.notna(row[title]) and str(row[title]).strip() != "":
            v = row[title]
        return v

    # ensure columns exist
    for c in ["dish","user","rating","Dish","User","Rating"]:
        if c not in df.columns:
            df[c] = pd.NA

    # build cleaned DataFrame
    cleaned = pd.DataFrame({
        "dish": df.apply(lambda r: coalesce_row(r, "dish", "Dish"), axis=1),
        "user": df.apply(lambda r: coalesce_row(r, "user", "User"), axis=1),
        "rating": df.apply(lambda r: coalesce_row(r, "rating", "Rating"), axis=1),
    })

    # normalize and convert types
    cleaned["dish"] = cleaned["dish"].astype(str).str.strip()
    cleaned["user"] = cleaned["user"].astype(str).str.strip()
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce").fillna(0).astype(float)

    # Keep last entry per (dish,user)
    cleaned = cleaned.drop_duplicates(subset=["dish","user"], keep="last").reset_index(drop=True)

    # backup original once
    backup = PATH + ".bak"
    if not os.path.exists(backup):
        os.rename(PATH, backup)
        print("Backup saved to", backup)
    else:
        print("Backup already exists:", backup)

    cleaned.to_csv(PATH, index=False)
    print("Cleaned file written to", PATH)
    print("Cleaned head:")
    print(cleaned.head(20))
