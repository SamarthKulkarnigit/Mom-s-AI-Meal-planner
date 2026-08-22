from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
import string
import random

from .database import engine, Base, get_db
from . import models

# Initialize database tables
Base.metadata.create_all(bind=engine)

# --- CONFIGURATION ---
import os
SECRET_KEY = os.getenv("SECRET_KEY", "your_super_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

app = FastAPI(title="Meal Planner API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- HELPER FUNCTIONS ---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def generate_group_code(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

# --- DEPENDENCY: GET CURRENT USER ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        group_id: str = payload.get("groupId")
        if username is None or group_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- REQUEST SCHEMAS ---
class CreateFamilyRequest(BaseModel):
    family_name: str
    creator_name: str
    password: str

class JoinFamilyRequest(BaseModel):
    group_code: str
    username: str
    password: str

# --- ROUTES ---

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

    new_group = models.Group(
        group_code=code,
        family_name=request.family_name,
        creator=request.creator_name
    )
    db.add(new_group)
    db.commit()
    db.refresh(new_group)

    new_user = models.User(
        username=request.creator_name,
        hashed_password=hash_password(request.password),
        group_code=code
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"group_code": code, "message": "Family created successfully"}

@app.post("/join_family")
def join_family(request: JoinFamilyRequest, db: Session = Depends(get_db)):
    if not db.query(models.Group).filter(models.Group.group_code == request.group_code).first():
        raise HTTPException(status_code=404, detail="Invalid family code")

    if db.query(models.User).filter(models.User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    new_user = models.User(
        username=request.username,
        hashed_password=hash_password(request.password),
        group_code=request.group_code
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

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

    access_token = create_access_token(
        data={"sub": user.username, "groupId": user.group_code}
    )
    return {"access_token": access_token, "token_type": "bearer", "group_code": user.group_code}

@app.get("/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"username": current_user.username, "group_code": current_user.group_code}

@app.get("/group/{group_code}/members")
def get_group_members(group_code: str, db: Session = Depends(get_db)):
    group = db.query(models.Group).filter(models.Group.group_code == group_code).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    users = db.query(models.User).filter(models.User.group_code == group_code).all()
    return [{"username": u.username} for u in users]

@app.get("/group/{group_code}/stats")
def get_group_stats(group_code: str, db: Session = Depends(get_db)):
    total_dishes = db.query(models.Dish).filter(models.Dish.group_code == group_code).count()

    ratings = db.query(models.Rating).join(models.Dish).filter(models.Dish.group_code == group_code).all()
    avg_rating = round(sum(r.rating for r in ratings) / len(ratings), 2) if ratings else 0.0

    best_dish = "N/A"
    if ratings:
        from collections import defaultdict
        dish_totals = defaultdict(list)
        for r in ratings:
            dish_totals[r.dish_id].append(r.rating)
        best_id = max(dish_totals, key=lambda d: sum(dish_totals[d]) / len(dish_totals[d]))
        best = db.query(models.Dish).filter(models.Dish.id == best_id).first()
        if best:
            best_dish = best.name

    schedule = db.query(models.ScheduleEntry).filter(
        models.ScheduleEntry.group_code == group_code
    ).order_by(models.ScheduleEntry.scheduled_date).limit(7).all()
    schedule_items = [
        {"dish": s.dish.name, "date": str(s.scheduled_date.date())}
        for s in schedule
    ]

    return {
        "total_dishes": total_dishes,
        "avg_rating": avg_rating,
        "best_dish": best_dish,
        "fatigue_dish": "N/A",
        "schedule": schedule_items,
    }
