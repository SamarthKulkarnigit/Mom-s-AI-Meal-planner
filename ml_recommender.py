# ml_recommender.py
import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import db

# ---------- small helpers ----------
def _find_column(df, candidates):
    """Find first matching column name from candidates (case-insensitive, partial match allowed)."""
    if df is None or df.empty:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for col_low, col_orig in cols_lower.items():
        for cand in candidates:
            if cand.lower() in col_low:
                return col_orig
    return None

def _ensure_series(df, possible_names):
    """
    Return a cleaned Series for the first matching column from possible_names.
    If the column is a DataFrame (e.g. due to duplicate names), collapse to first column.
    Always returns a string Series (trimmed).
    """
    col = _find_column(df, possible_names)
    if col is None:
        return pd.Series([], dtype=str)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(str).fillna("").str.strip()

def _norm(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if s.empty or s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# ---------- time decay weighted ratings ----------
TIME_DECAY_ALPHA = 0.8
# closer to 1.0 = slower forgetting
# smaller = faster adaptation

def _compute_time_weighted_ratings(ratings_df):
    """
    Computes time-weighted average ratings.
    Recent weeks matter more.
    """

    if ratings_df is None or ratings_df.empty:
        return pd.DataFrame(columns=["dish", "avg_rating", "n_ratings"])

    df = ratings_df.copy()

    # ensure week exists
    if "week" not in df.columns:
        df["week"] = 1

    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(1).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    current_week = df["week"].max()

    # weeks ago
    df["weeks_ago"] = current_week - df["week"]

    # exponential decay
    df["weight"] = TIME_DECAY_ALPHA ** df["weeks_ago"]

    # weighted rating
    df["weighted_rating"] = df["rating"] * df["weight"]

    # aggregate
    grouped = df.groupby("dish").agg(
        weighted_sum=("weighted_rating", "sum"),
        weight_sum=("weight", "sum"),
        n_ratings=("rating", "count")
    ).reset_index()

    grouped["avg_rating"] = grouped["weighted_sum"] / grouped["weight_sum"]

    return grouped[["dish", "avg_rating", "n_ratings"]]

# ---------- debug logger ----------
def _log(*args):
    # print to stderr so Streamlit UI doesn't swallow it
    print(*args, file=sys.stderr)

# ---------- loaders ----------
def _load_clean_data(group_code: str):
    """Return ratings, polls, dishes, group_df (all DataFrames). Defensive and prints debug."""
    # read raw using db.load_data (app view)
    r = db.load_data(f"ratings_{group_code}.csv")
    if r is None:
        r = pd.DataFrame()
    p = db.load_data(f"poll_{group_code}.csv")
    if p is None:
        p = pd.DataFrame()
    d = db.load_data(f"dishes_{group_code}.csv")
    if d is None:
        d = pd.DataFrame()
    g = db.load_data(f"group_{group_code}.csv")
    if g is None:
        g = pd.DataFrame()

    # Debug print: shapes and columns
    _log("DEBUG: raw shapes -> ratings", getattr(r,'shape',None), "polls", getattr(p,'shape',None), "dishes", getattr(d,'shape',None))
    _log("DEBUG: ratings cols:", list(r.columns))
    _log("DEBUG: polls cols:", list(p.columns))
    _log("DEBUG: dishes cols:", list(d.columns))

    # ---------------------
    # Normalize ratings (defensive)
    # ---------------------
    if not r.empty:
        # sanitize column headers (strip whitespace)
        r_cols = [c.strip() for c in r.columns]
        r.columns = r_cols

        # Use helper to coalesce any variants (Dish / dish / Dish ) into clean Series
        s_dish = _ensure_series(r, ["dish", "name"])
        s_user = _ensure_series(r, ["user", "username", "member"])

        # rating may be numeric or stored as text; try to find numeric-like column name
        rating_col = _find_column(r, ["rating", "score", "stars"])
        if rating_col is not None:
            s_rating = r[rating_col]
            if isinstance(s_rating, pd.DataFrame):
                s_rating = s_rating.iloc[:, 0]
            s_rating = pd.to_numeric(s_rating, errors="coerce").fillna(0.0).astype(float)
        else:
            # fallback: if single-column file (only one column), treat it as dish names
            if len(r_cols) == 1:
                s_dish = r.iloc[:, 0].astype(str).fillna("").str.strip()
                s_user = pd.Series([""] * len(s_dish))
                s_rating = pd.Series([0.0] * len(s_dish))
            else:
                s_rating = pd.Series([0.0] * len(s_dish))

        # parse week if it exists
        week_col = _find_column(r, ["week", "week_num"])
        if week_col is not None:
            s_week = r[week_col]
            if isinstance(s_week, pd.DataFrame):
                s_week = s_week.iloc[:, 0]
            s_week = pd.to_numeric(s_week, errors="coerce").fillna(1).astype(int)
        else:
            s_week = pd.Series([1] * len(s_dish))

        # Build normalized DataFrame with guaranteed Series types
        r = pd.DataFrame({
            "dish": s_dish.astype(str).fillna("").str.strip(),
            "user": s_user.astype(str).fillna("").str.strip(),
            "rating": s_rating.astype(float),
            "week": s_week.astype(int)
        })
    else:
        r = pd.DataFrame(columns=["dish","user","rating","week"])

    # ---------------------
    # Normalize polls
    # ---------------------
    if not p.empty:
        p_cols = [c.strip() for c in p.columns]
        p.columns = p_cols
        poll_dish = _find_column(p, ["dish","name","poll"])
        votes_col = _find_column(p, ["votes","vote","count"])
        if poll_dish:
            p = p.rename(columns={poll_dish: "dish"})
        else:
            p["dish"] = ""
        if votes_col:
            p = p.rename(columns={votes_col: "votes"})
        else:
            p["votes"] = 0
        p = p[[c for c in ["dish","votes"] if c in p.columns]]
        if "dish" in p.columns:
            # use _ensure_series approach only for reading; then assign back as column
            p["dish"] = _ensure_series(p, ["dish"])
        if "votes" in p.columns:
            p["votes"] = pd.to_numeric(p["votes"], errors="coerce").fillna(0).astype(int)
    else:
        p = pd.DataFrame(columns=["dish","votes"])

    # ---------------------
    # Normalize dishes
    # ---------------------
    if not d.empty:
        d_cols = [c.strip() for c in d.columns]
        d.columns = d_cols
        dish_col = _find_column(d, ["dish","name"])
        if dish_col:
            d = d.rename(columns={dish_col: "dish"})
            d = d[["dish"]]
            d["dish"] = _ensure_series(d, ["dish"])
        else:
            # fallback: if single column, use it
            if len(d_cols) == 1:
                d = pd.DataFrame({"dish": d.iloc[:,0].astype(str).fillna("").str.strip()})
            else:
                d = pd.DataFrame(columns=["dish"])
    else:
        d = pd.DataFrame(columns=["dish"])

    # drop empty-dish rows
    r = r[r["dish"].astype(str).str.strip() != ""].reset_index(drop=True) if not r.empty else r
    p = p[p["dish"].astype(str).str.strip() != ""].reset_index(drop=True) if not p.empty else p
    d = d[d["dish"].astype(str).str.strip() != ""].drop_duplicates().reset_index(drop=True) if not d.empty else d

    # debug after cleanup
    _log("DEBUG after-clean shapes -> ratings", getattr(r,'shape',None), "polls", getattr(p,'shape',None), "dishes", getattr(d,'shape',None))
    _log("DEBUG after-clean ratings cols:", list(r.columns))
    return r, p, d, g

# ---------- similarity helpers ----------
def _build_content_sim(dishes: pd.DataFrame):
    if dishes is None or dishes.empty: return pd.DataFrame()
    texts = dishes["dish"].astype(str).tolist()
    try:
        tfidf = TfidfVectorizer()
        mat = tfidf.fit_transform(texts)
        sim = cosine_similarity(mat)
        return pd.DataFrame(sim, index=dishes["dish"], columns=dishes["dish"])
    except Exception as e:
        _log("content sim error:", e)
        return pd.DataFrame()

def _build_cf_sim(ratings: pd.DataFrame):
    if ratings is None or ratings.empty: return pd.DataFrame()
    try:
        rm = ratings.pivot_table(index="user", columns="dish", values="rating", fill_value=0.0)
        if rm.empty: return pd.DataFrame()
        sim = cosine_similarity(rm.T)
        return pd.DataFrame(sim, index=rm.columns, columns=rm.columns)
    except Exception as e:
        _log("cf sim error:", e)
        return pd.DataFrame()

# ---------- main planner ----------
def generate_weekly_plan_for_group(group_code: str, n_days: int = 7, return_stats: bool = False):
    _log(f"AI: generate_weekly_plan_for_group({group_code}) start")
    ratings, polls, dishes, group_df = _load_clean_data(group_code)

    _log("AI: shapes -> ratings", getattr(ratings,'shape',None), "polls", getattr(polls,'shape',None), "dishes", getattr(dishes,'shape',None))

   # ---- WEEK-AWARE RATING STATISTICS ----
# If ratings contain a Week column, apply time decay:
#   recent week   → high weight
#   older weeks   → low weight
# If no Week column exists, fallback to simple mean.

    if ratings is None or ratings.empty:
        rating_stats = pd.DataFrame(columns=["dish","avg_rating","n_ratings"])
    else:
        # try to detect week column
        week_col = None
        for c in ratings.columns:
            if c.lower().strip() == "week":
                week_col = c
                break

        if week_col is None:
            # fallback → normal average
            rating_stats = ratings.groupby("dish").agg(
                avg_rating=("rating", "mean"),
                n_ratings=("rating", "count")
            ).reset_index()

        else:
            # --- TIME DECAY: weight = 1 / (1 + (max_week - week)) ---
            ratings = ratings.copy()
            ratings["week_numeric"] = pd.to_numeric(ratings[week_col], errors="coerce").fillna(0).astype(int)
            max_w = ratings["week_numeric"].max()

            # exponential decay is smoother & well-behaved
            #   most recent → weight = 1.0
            #   each older week reduces weight by ~20%
            ratings["weight"] = 0.8 ** (max_w - ratings["week_numeric"])

            rating_stats = ratings.groupby("dish").apply(
                lambda g: pd.Series({
                    "avg_rating": np.average(g["rating"], weights=g["weight"]),
                    "n_ratings": len(g)
                })
            ).reset_index()

    dish_stats = dishes[["dish"]].drop_duplicates().copy() if not dishes.empty else pd.DataFrame(columns=["dish"])
    try:
        dish_stats = dish_stats.merge(rating_stats, on="dish", how="left").merge(polls[["dish","votes"]], on="dish", how="left")
    except Exception as e:
        _log("merge error building dish_stats:", e)
        # ensure columns
        if "avg_rating" not in dish_stats.columns: dish_stats["avg_rating"] = 0.0
        if "n_ratings" not in dish_stats.columns: dish_stats["n_ratings"] = 0.0
        if "votes" not in dish_stats.columns: dish_stats["votes"] = 0.0

    if dish_stats.empty:
        _log("AI: dish_stats empty -> returning None")
        return None

    # sanitize numeric columns
    dish_stats["avg_rating"] = pd.to_numeric(dish_stats.get("avg_rating", 0), errors="coerce").fillna(0.0).astype(float)
    dish_stats["votes"] = pd.to_numeric(dish_stats.get("votes", 0), errors="coerce").fillna(0.0).astype(float)

    # build sims
    content_sim = _build_content_sim(dishes)
    cf_sim = _build_cf_sim(ratings)

    # scoring
    dish_stats["rating_score"] = _norm(dish_stats["avg_rating"])
    dish_stats["popularity_score"] = _norm(dish_stats["votes"])

    cf_scores = []
    content_scores = []
    for dish in dish_stats["dish"].tolist():
        cf_scores.append(0.0)
        content_scores.append(0.0)
        if not cf_sim.empty and dish in cf_sim.index:
            # similarity to top-rated dishes
            cf_scores[-1] = float(cf_sim.loc[dish].max()) if not cf_sim.loc[dish].empty else 0.0
        if not content_sim.empty and dish in content_sim.index:
            content_scores[-1] = float(content_sim.loc[dish].max()) if not content_sim.loc[dish].empty else 0.0

    dish_stats["cf_score"] = _norm(pd.Series(cf_scores))
    dish_stats["content_score"] = _norm(pd.Series(content_scores))

    dish_stats["hybrid_score"] = (
        0.40 * dish_stats["rating_score"]
        + 0.25 * dish_stats["popularity_score"]
        + 0.20 * dish_stats["cf_score"]
        + 0.15 * dish_stats["content_score"]
    )

    dish_stats = dish_stats.sort_values(
    "hybrid_score",
    ascending=False
).reset_index(drop=True)

    # --------------------------------
    # CONTROLLED RANDOMNESS
    # --------------------------------

    top_k = min(20, len(dish_stats))

    top_pool = dish_stats.head(top_k).copy()

    # weighted randomization
    # add an epsilon to ensure all probabilities are non-zero and well-behaved (prevents ValueError when replace=False in sample)
    scores_epsilon = top_pool["hybrid_score"] + 0.1
    top_pool["probability"] = (
        scores_epsilon
        / scores_epsilon.sum()
    )

    # A-Res (Adjusted Reservoir Sampling) algorithm for mathematically correct weighted sampling without replacement
    try:
        # Generate random keys: u^(1/w)
        random_keys = np.random.rand(len(top_pool)) ** (1.0 / top_pool["probability"])
        # Sort candidates by keys in descending order
        sorted_indices = np.argsort(-random_keys)
        candidates = top_pool.iloc[sorted_indices]["dish"].tolist()
    except Exception as e:
        _log("WARNING: Weighted permutation sampling failed, falling back to uniform sampling:", e)
        candidates = (
            top_pool.sample(
                frac=1,
                weights=None,
                random_state=None
            )["dish"]
            .tolist()
        )

    # pick non-similar set
    chosen = []
    for dish in candidates:
        if len(chosen) >= n_days:
            break
        too_similar = False
        for c in chosen:
            sim_cf = cf_sim.loc[dish, c] if (not cf_sim.empty and dish in cf_sim.index and c in cf_sim.columns) else 0.0
            sim_content = content_sim.loc[dish, c] if (not content_sim.empty and dish in content_sim.index and c in content_sim.columns) else 0.0
            if max(sim_cf, sim_content) >= 0.9:
                too_similar = True
                break
        if not too_similar:
            chosen.append(dish)

    # pad if needed
    if len(chosen) < n_days:
        for dname in candidates:
            if len(chosen) >= n_days: break
            if dname not in chosen:
                chosen.append(dname)

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    plan = pd.DataFrame({"Day": days[:n_days], "Dish": chosen[:n_days]})
    _log("AI: returning plan with rows:", len(plan))
    if return_stats:
        return plan, dish_stats
    return plan


# ---------- candidate export for Gemini ----------
def get_candidates_for_group(group_code: str, n_candidates: int = 14) -> list:
    """
    Return the top n_candidates ranked meals for a group as structured dicts
    suitable for passing to the LLM service.

    Each dict:
    {
        "dish_id":              int,   # DB ID
        "dish_name":            str,
        "recommendation_score": float, # hybrid 0-1
        "evidence":             list[str]  # grounded factual signals
    }

    Reuses the existing scoring pipeline — no duplicate math.
    Returns [] if insufficient data.
    """
    import db as _db

    _log(f"CANDIDATES: get_candidates_for_group({group_code}, n={n_candidates})")
    ratings, polls, dishes, group_df = _load_clean_data(group_code)

    if dishes is None or dishes.empty:
        _log("CANDIDATES: no dishes — returning empty")
        return []

    # --- time-decayed rating stats ---
    if ratings is None or ratings.empty:
        rating_stats = pd.DataFrame(columns=["dish", "avg_rating", "n_ratings"])
    else:
        week_col = next((c for c in ratings.columns if c.lower().strip() == "week"), None)
        if week_col is None:
            rating_stats = ratings.groupby("dish").agg(
                avg_rating=("rating", "mean"),
                n_ratings=("rating", "count"),
            ).reset_index()
        else:
            ratings = ratings.copy()
            ratings["week_numeric"] = pd.to_numeric(ratings[week_col], errors="coerce").fillna(0).astype(int)
            max_w = ratings["week_numeric"].max()
            ratings["weight"] = 0.8 ** (max_w - ratings["week_numeric"])
            rating_stats = ratings.groupby("dish").apply(
                lambda g: pd.Series({
                    "avg_rating": np.average(g["rating"], weights=g["weight"]),
                    "n_ratings": len(g),
                })
            ).reset_index()

    dish_stats = dishes[["dish"]].drop_duplicates().copy()
    try:
        dish_stats = (
            dish_stats
            .merge(rating_stats, on="dish", how="left")
            .merge(polls[["dish", "votes"]], on="dish", how="left")
        )
    except Exception:
        if "avg_rating" not in dish_stats.columns:
            dish_stats["avg_rating"] = 0.0
        if "n_ratings" not in dish_stats.columns:
            dish_stats["n_ratings"] = 0.0
        if "votes" not in dish_stats.columns:
            dish_stats["votes"] = 0.0

    dish_stats["avg_rating"] = pd.to_numeric(dish_stats.get("avg_rating", 0), errors="coerce").fillna(0.0)
    dish_stats["votes"] = pd.to_numeric(dish_stats.get("votes", 0), errors="coerce").fillna(0.0)

    content_sim = _build_content_sim(dishes)
    cf_sim = _build_cf_sim(ratings)

    dish_stats["rating_score"] = _norm(dish_stats["avg_rating"])
    dish_stats["popularity_score"] = _norm(dish_stats["votes"])

    cf_scores, content_scores = [], []
    for dish in dish_stats["dish"].tolist():
        cf = float(cf_sim.loc[dish].max()) if (not cf_sim.empty and dish in cf_sim.index) else 0.0
        ct = float(content_sim.loc[dish].max()) if (not content_sim.empty and dish in content_sim.index) else 0.0
        cf_scores.append(cf)
        content_scores.append(ct)

    dish_stats["cf_score"] = _norm(pd.Series(cf_scores))
    dish_stats["content_score"] = _norm(pd.Series(content_scores))

    dish_stats["hybrid_score"] = (
        0.40 * dish_stats["rating_score"]
        + 0.25 * dish_stats["popularity_score"]
        + 0.20 * dish_stats["cf_score"]
        + 0.15 * dish_stats["content_score"]
    )

    dish_stats = dish_stats.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    top = dish_stats.head(n_candidates)

    # --- fetch dish IDs from SQLAlchemy (needed for validation) ---
    try:
        from backend.database import SessionLocal
        from backend import models as _models
        db_sess = SessionLocal()
        dish_id_map = {
            d.name.lower(): d.id
            for d in db_sess.query(_models.Dish).filter(_models.Dish.group_code == group_code).all()
        }
        db_sess.close()
    except Exception as e:
        _log("CANDIDATES: could not load dish IDs from DB:", e)
        dish_id_map = {}

    candidates = []
    for _, row in top.iterrows():
        dish_name = str(row["dish"])
        dish_id = dish_id_map.get(dish_name.lower())
        if dish_id is None:
            continue  # skip dishes not in DB — safety filter

        evidence = []
        if float(row.get("rating_score", 0)) > 0.5:
            evidence.append("high family rating")
        if float(row.get("popularity_score", 0)) > 0.5:
            evidence.append("highly voted in polls")
        if float(row.get("cf_score", 0)) > 0.5:
            evidence.append("matches preferences of similar family members")
        if not evidence:
            evidence.append("adds variety to the weekly rotation")

        candidates.append({
            "dish_id": dish_id,
            "dish_name": dish_name,
            "recommendation_score": round(float(row["hybrid_score"]), 4),
            "evidence": evidence,
        })

    _log(f"CANDIDATES: returning {len(candidates)} candidates")
    return candidates