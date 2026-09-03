"""
backend/main.py

FastAPI backend for the Meal Planner application.

Phase 2 additions:
  POST /group/{group_code}/schedule/generate
    - Retrieves ML candidates, calls Gemini, validates strictly, persists atomically.
    - Falls back to deterministic ML plan on any LLM failure.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List
from collections import defaultdict
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
import string
import random
import logging
import time
import sys
import os

from .database import engine, Base, get_db, run_schema_migrations
from . import models

logger = logging.getLogger(__name__)

# ─── Bootstrap ────────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)
run_schema_migrations()  # additive, idempotent; no-op on fresh databases

SECRET_KEY = os.getenv("SECRET_KEY", "your_super_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

app = FastAPI(title="Meal Planner API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ─── Auth helpers ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def generate_group_code(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        group_id: str = payload.get("groupId")
        if username is None or group_id is None:
            raise exc
    except JWTError:
        raise exc

    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise exc
    return user


# ─── Request schemas ──────────────────────────────────────────────────────────

class CreateFamilyRequest(BaseModel):
    family_name: str
    creator_name: str
    password: str


class JoinFamilyRequest(BaseModel):
    group_code: str
    username: str
    password: str


# ─── Core routes ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/create_family")
def create_family(request: CreateFamilyRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == request.creator_name).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    code = generate_group_code()
    while db.query(models.Group).filter(models.Group.group_code == code).first():
        code = generate_group_code()

    db.add(models.Group(group_code=code, family_name=request.family_name, creator=request.creator_name))
    db.commit()
    db.add(models.User(username=request.creator_name, hashed_password=hash_password(request.password), group_code=code))
    db.commit()
    return {"group_code": code, "message": "Family created successfully"}


@app.post("/join_family")
def join_family(request: JoinFamilyRequest, db: Session = Depends(get_db)):
    if not db.query(models.Group).filter(models.Group.group_code == request.group_code).first():
        raise HTTPException(status_code=404, detail="Invalid family code")
    if db.query(models.User).filter(models.User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    db.add(models.User(username=request.username, hashed_password=hash_password(request.password), group_code=request.group_code))
    db.commit()
    return {"message": "Joined family successfully"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username, "groupId": user.group_code})
    return {"access_token": token, "token_type": "bearer", "group_code": user.group_code}


@app.get("/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"username": current_user.username, "group_code": current_user.group_code}


def _ensure_group_access(current_user: models.User, group_code: str) -> None:
    """Raise 403 unless the authenticated user belongs to the requested group."""
    if current_user.group_code != group_code:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not belong to this group",
        )


@app.get("/group/{group_code}/members")
def get_group_members(
    group_code: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_group_access(current_user, group_code)
    group = db.query(models.Group).filter(models.Group.group_code == group_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    users = db.query(models.User).filter(models.User.group_code == group_code).all()
    return [{"username": u.username} for u in users]


@app.get("/group/{group_code}/stats")
def get_group_stats(
    group_code: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_group_access(current_user, group_code)
    group = db.query(models.Group).filter(models.Group.group_code == group_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    total_dishes = db.query(models.Dish).filter(models.Dish.group_code == group_code).count()

    ratings = db.query(models.Rating).join(models.Dish).filter(models.Dish.group_code == group_code).all()
    avg_rating = round(sum(r.rating for r in ratings) / len(ratings), 2) if ratings else 0.0

    best_dish = "N/A"
    if ratings:
        dish_totals = defaultdict(list)
        for r in ratings:
            dish_totals[r.dish_id].append(r.rating)
        best_id = max(dish_totals, key=lambda d: sum(dish_totals[d]) / len(dish_totals[d]))
        best = db.query(models.Dish).filter(models.Dish.id == best_id).first()
        if best:
            best_dish = best.name

    # Current planning week only (latest week with a saved plan, else 1), in
    # Monday-Sunday order. "date" is kept for backward compatibility and holds
    # the weekday name.
    max_week = db.query(func.max(models.ScheduleEntry.week)).filter(
        models.ScheduleEntry.group_code == group_code
    ).scalar()
    stats_week = max_week or 1

    schedule = (
        db.query(models.ScheduleEntry)
        .filter(
            models.ScheduleEntry.group_code == group_code,
            models.ScheduleEntry.week == stats_week,
        )
        .all()
    )
    day_order = {d: i for i, d in enumerate(VALID_DAYS)}
    schedule.sort(key=lambda s: day_order.get(s.day, len(VALID_DAYS)))
    schedule_items = [
        {"dish": s.dish.name if s.dish else "Unknown", "date": s.day}
        for s in schedule
    ]

    return {
        "total_dishes": total_dishes,
        "avg_rating": avg_rating,
        "best_dish": best_dish,
        "fatigue_dish": "N/A",
        "week": stats_week,
        "schedule": schedule_items,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY PLAN GENERATION  –  POST /group/{group_code}/schedule/generate
# ─────────────────────────────────────────────────────────────────────────────

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_CANDIDATES_REQUIRED = 7


def _resolve_target_week(db: Session, group_code: str) -> int:
    """
    Week semantics — single source of truth (mirrored in db.get_current_planning_week):

    - No plan exists yet: the next logical week (1).
    - A plan already exists for the latest week: that same week, so that
      regenerating the current week's plan replaces it in place.

    The UI and the persisted rows both agree because both resolve from the
    same rows in ScheduleEntry.
    """
    max_week = db.query(func.max(models.ScheduleEntry.week)).filter(
        models.ScheduleEntry.group_code == group_code
    ).scalar()
    return max_week or 1


def _validate_llm_plan(
    plan_entries: list,
    candidate_ids: set,
    group_dish_ids: set,
) -> List[dict]:
    """
    Strict authoritative backend validation of the LLM-produced plan.
    Raises ValueError on any violation. The caller decides fallback behaviour.
    """
    if not isinstance(plan_entries, list):
        raise ValueError(f"Plan must be a list, got {type(plan_entries).__name__}")
    if len(plan_entries) != 7:
        raise ValueError(f"Plan must have exactly 7 entries, got {len(plan_entries)}")

    seen_days: set = set()
    seen_dish_ids: set = set()
    validated: List[dict] = []

    for i, entry in enumerate(plan_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} is not a dict: {entry!r}")

        missing = [k for k in ("day", "dish_id", "reason") if k not in entry]
        if missing:
            raise ValueError(f"Entry {i} missing keys: {missing}")

        day = entry["day"]
        if day not in VALID_DAYS:
            raise ValueError(f"Entry {i} has invalid day '{day}'")
        if day in seen_days:
            raise ValueError(f"Duplicate day '{day}' in LLM plan")
        seen_days.add(day)

        try:
            dish_id = int(entry["dish_id"])
        except (TypeError, ValueError):
            raise ValueError(f"Entry {i} dish_id '{entry['dish_id']}' is not an integer")

        if dish_id not in group_dish_ids:
            raise ValueError(f"Entry {i} dish_id={dish_id} does not belong to this group")
        if dish_id not in candidate_ids:
            raise ValueError(f"Entry {i} dish_id={dish_id} was not in the candidate set sent to Gemini")
        if dish_id in seen_dish_ids:
            raise ValueError(f"Entry {i} dish_id={dish_id} appears more than once")
        seen_dish_ids.add(dish_id)

        reason = str(entry.get("reason", "")).strip()
        if len(reason) < 5:
            raise ValueError(f"Entry {i} reason is too short: '{reason}'")

        validated.append({"day": day, "dish_id": dish_id, "reason": reason})

    if set(e["day"] for e in validated) != set(VALID_DAYS):
        raise ValueError("Plan does not cover all 7 days of the week")

    return validated


def _fallback_plan_entries(group_code: str, db: Session) -> List[dict]:
    """
    Build a 7-day plan using the deterministic ML recommender.
    Returns a list of entry dicts in the same shape as the AI path.
    Raises HTTPException if there are not enough dishes.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ml_recommender import generate_weekly_plan_for_group
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"ML recommender unavailable: {exc}")

    plan_df = generate_weekly_plan_for_group(group_code)
    if plan_df is None or plan_df.empty:
        raise HTTPException(status_code=422, detail="Not enough dishes to generate a plan.")

    dish_map = {
        d.name.lower(): d.id
        for d in db.query(models.Dish).filter(models.Dish.group_code == group_code).all()
    }

    entries = []
    for _, row in plan_df.iterrows():
        dish_name = str(row.get("Dish", "")).strip()
        dish_id = dish_map.get(dish_name.lower())
        if dish_id is None:
            continue
        entries.append({
            "day": str(row.get("Day", "")),
            "dish_id": dish_id,
            "dish_name": dish_name,
            "reason": "Selected by the recommendation engine based on family preferences and variety.",
        })

    if len(entries) < 7:
        raise HTTPException(
            status_code=422,
            detail="Not enough dishes could be mapped to a 7-day plan."
        )
    return entries


def _persist_plan(group_code: str, week: int, plan_entries: List[dict], db: Session) -> None:
    """Atomically replace any existing schedule for (group, week) with the new plan."""
    db.query(models.ScheduleEntry).filter(
        models.ScheduleEntry.group_code == group_code,
        models.ScheduleEntry.week == week,
    ).delete(synchronize_session=False)

    # Insert in canonical Monday-Sunday order so row order is stable.
    day_order = {d: i for i, d in enumerate(VALID_DAYS)}
    ordered = sorted(plan_entries, key=lambda e: day_order.get(e["day"], len(VALID_DAYS)))

    now = datetime.utcnow()
    for entry in ordered:
        db.add(models.ScheduleEntry(
            group_code=group_code,
            dish_id=entry["dish_id"],
            week=week,
            day=entry["day"],
            scheduled_date=now,
            reason=entry.get("reason") or None,
        ))
    db.commit()
    logger.info("SCHEDULE: persisted %d entries group=%s week=%d", len(plan_entries), group_code, week)


@app.post("/group/{group_code}/schedule/generate")
def generate_schedule(
    group_code: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    AI-assisted weekly meal plan generation.

    1. Auth: verify user belongs to the group.
    2. Candidates: ML recommender produces top-14 scored dishes.
    3. LLM: Gemini selects 7 and writes grounded reasons.
    4. Validation: strict backend checks (schema, days, dish IDs, group ownership).
    5. Fallback: deterministic ML plan on any LLM failure.
    6. Persistence: atomic replace of ScheduleEntry rows (incl. reasons).
    7. Response: week + schedule (day, dish, reason) + ai_generated/fallback_used.
    """
    t_start = time.time()
    logger.info("SCHEDULE/GENERATE: user=%s group=%s", current_user.username, group_code)

    # 1. Auth
    _ensure_group_access(current_user, group_code)

    group = db.query(models.Group).filter(models.Group.group_code == group_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    # 2. Determine target week (regenerate the latest plan week, or start week 1)
    target_week = _resolve_target_week(db, group_code)

    # 3. Candidate generation
    t0 = time.time()
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ml_recommender import get_candidates_for_group
        candidates = get_candidates_for_group(group_code, n_candidates=14)
    except Exception as exc:
        logger.error("SCHEDULE/GENERATE: candidate generation error: %s", exc)
        raise HTTPException(status_code=503, detail=f"Recommendation engine error: {exc}")

    logger.info("SCHEDULE/GENERATE: candidates ready in %.2fs (n=%d)", time.time() - t0, len(candidates))

    if len(candidates) < MIN_CANDIDATES_REQUIRED:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough dishes to build a 7-day plan "
                f"({len(candidates)} available, need {MIN_CANDIDATES_REQUIRED}). "
                f"Add more dishes via the Polls page."
            ),
        )

    candidate_ids = {c["dish_id"] for c in candidates}
    group_dish_ids = {d.id for d in db.query(models.Dish).filter(models.Dish.group_code == group_code).all()}

    # Build lean family context (no secrets, no PII beyond names)
    members = [m.name for m in db.query(models.Member).filter(models.Member.group_code == group_code).all()]
    if not members:
        members = [u.username for u in db.query(models.User).filter(models.User.group_code == group_code).all()]

    recent_served = [
        sl.dish.name
        for sl in (
            db.query(models.ServedLog)
            .filter(models.ServedLog.group_code == group_code)
            .order_by(models.ServedLog.id.desc())
            .limit(14)
            .all()
        )
    ]

    family_context = {
        "family_name": group.family_name,
        "members": members,
        "recent_dishes": recent_served,
    }

    # 4. LLM path with fallback
    ai_generated = True
    fallback_reason: Optional[str] = None
    plan_entries: Optional[List[dict]] = None

    t1 = time.time()
    try:
        from .llm_service import generate_weekly_plan as llm_generate, LLMServiceError
        raw_llm = llm_generate(candidates, family_context)
        logger.info("SCHEDULE/GENERATE: Gemini responded in %.2fs", time.time() - t1)

        validated = _validate_llm_plan(raw_llm, candidate_ids, group_dish_ids)

        dish_id_to_name = {c["dish_id"]: c["dish_name"] for c in candidates}
        plan_entries = [
            {**e, "dish_name": dish_id_to_name.get(e["dish_id"], "Unknown")}
            for e in validated
        ]

    except Exception as exc:
        logger.warning(
            "SCHEDULE/GENERATE: AI path failed (%s: %s) — using ML fallback",
            type(exc).__name__, exc,
        )
        ai_generated = False
        fallback_reason = str(exc)
        plan_entries = _fallback_plan_entries(group_code, db)

    # 5. Persist atomically
    _persist_plan(group_code, target_week, plan_entries, db)

    logger.info(
        "SCHEDULE/GENERATE: done in %.2fs ai=%s week=%d",
        time.time() - t_start, ai_generated, target_week,
    )

    # Response in canonical Monday-Sunday order
    day_order = {d: i for i, d in enumerate(VALID_DAYS)}
    ordered_entries = sorted(plan_entries, key=lambda e: day_order.get(e["day"], len(VALID_DAYS)))

    response: dict = {
        "week": target_week,
        "ai_generated": ai_generated,
        "fallback_used": not ai_generated,
        "schedule": [
            {
                "day": e["day"],
                "dish_id": e["dish_id"],
                "dish_name": e.get("dish_name", ""),
                "reason": e.get("reason"),
            }
            for e in ordered_entries
        ],
    }

    if not ai_generated:
        response["fallback_notice"] = (
            "AI planning is temporarily unavailable. "
            "We generated a recommendation-based plan instead."
        )

    return response


@app.get("/group/{group_code}/schedule")
def get_group_schedule(
    group_code: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return the current (latest) saved weekly plan with persisted reasons,
    ordered Monday-Sunday. Empty schedule when the group has no plan yet.
    """
    _ensure_group_access(current_user, group_code)
    group = db.query(models.Group).filter(models.Group.group_code == group_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    week = _resolve_target_week(db, group_code)
    entries = (
        db.query(models.ScheduleEntry)
        .filter(
            models.ScheduleEntry.group_code == group_code,
            models.ScheduleEntry.week == week,
        )
        .all()
    )
    day_order = {d: i for i, d in enumerate(VALID_DAYS)}
    entries.sort(key=lambda e: day_order.get(e.day, len(VALID_DAYS)))

    return {
        "week": week,
        "schedule": [
            {
                "day": e.day,
                "dish_id": e.dish_id,
                "dish_name": e.dish.name if e.dish else "",
                "reason": e.reason,
            }
            for e in entries
        ],
    }
