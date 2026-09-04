from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from .database import Base
import datetime

# User model (Registered accounts)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    group = relationship("Group", back_populates="users")
    ratings = relationship("Rating", back_populates="user")

# Group model (Maps to groups.csv)
class Group(Base):
    __tablename__ = "groups"
    group_code = Column(String, primary_key=True, index=True)
    family_name = Column(String, nullable=False)
    creator = Column(String, nullable=False)
    users = relationship("User", back_populates="group")
    members = relationship("Member", back_populates="group")
    dishes = relationship("Dish", back_populates="group")
    polls = relationship("Poll", back_populates="group")
    schedule_entries = relationship("ScheduleEntry", back_populates="group")

# Member model (Maps to group_{code}.csv)
class Member(Base):
    __tablename__ = "members"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    name = Column(String, index=True, nullable=False)
    likes = Column(String, nullable=True)
    dislikes = Column(String, nullable=True)
    group = relationship("Group", back_populates="members")

    __table_args__ = (
        UniqueConstraint('group_code', 'name', name='uq_member_group_name'),
    )

# Dish model (Maps to dishes_{code}.csv)
class Dish(Base):
    __tablename__ = "dishes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    category = Column(String, nullable=True)
    source = Column(String, nullable=True, default="Poll")
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    group = relationship("Group", back_populates="dishes")
    ratings = relationship("Rating", back_populates="dish")
    poll_options = relationship("PollOption", back_populates="dish")
    schedule_entries = relationship("ScheduleEntry", back_populates="dish")

# Rating model (Maps to ratings_{code}.csv)
class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True) # Added for quick lookup
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Optional if rated by just member name
    user_name = Column(String, nullable=True) # Added to map 'user' column easily
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    rating = Column(Float, nullable=False)
    week = Column(Integer, nullable=True)
    day = Column(String, nullable=True)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="ratings")
    dish = relationship("Dish", back_populates="ratings")

    __table_args__ = (
        UniqueConstraint('group_code', 'dish_id', 'user_name', name='uq_rating_group_dish_user'),
    )

# Poll Vote model (Maps to poll_votes_{code}.csv)
class PollVote(Base):
    __tablename__ = "poll_votes"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    user_name = Column(String, nullable=False)
    vote = Column(Integer, default=1)

# Pending Suggestion model (Maps to pending_{code}.csv)
class PendingSuggestion(Base):
    __tablename__ = "pending_suggestions"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    dish_name = Column(String, nullable=False)
    suggester = Column(String, nullable=False)

# Served Log model (Maps to served_log_{code}.csv)
class ServedLog(Base):
    __tablename__ = "served_logs"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    day = Column(String, nullable=False)
    week = Column(Integer, nullable=False)
    # ORM relationship only (no schema change): lets code resolve the served
    # dish name via sl.dish.name, matching the convention used by ScheduleEntry,
    # Rating, and PollOption. Previously only dish_id existed, so any code path
    # that accessed sl.dish raised AttributeError.
    dish = relationship("Dish")

# ScheduleEntry model (Maps to schedule_{code}*.csv)
class ScheduleEntry(Base):
    __tablename__ = "schedule"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    scheduled_date = Column(DateTime, nullable=True)
    week = Column(Integer, nullable=False)
    day = Column(String, nullable=False)
    reason = Column(Text, nullable=True)  # grounded explanation (AI or recommender)
    group = relationship("Group", back_populates="schedule_entries")
    dish = relationship("Dish", back_populates="schedule_entries")

    __table_args__ = (
        UniqueConstraint('group_code', 'week', 'day', name='uq_schedule_group_week_day'),
    )

# Existing Poll models (Legacy/Extra, kept for compatibility if needed)
class Poll(Base):
    __tablename__ = "polls"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    group = relationship("Group", back_populates="polls")
    options = relationship("PollOption", back_populates="poll")

class PollOption(Base):
    __tablename__ = "poll_options"
    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, ForeignKey("polls.id"))
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    option_text = Column(String, nullable=False)
    votes = Column(Integer, default=0)
    poll = relationship("Poll", back_populates="options")
    dish = relationship("Dish", back_populates="poll_options")
