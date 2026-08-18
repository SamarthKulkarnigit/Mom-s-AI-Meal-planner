from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text
from sqlalchemy.orm import relationship
from .database import Base
import datetime

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    group = relationship("Group", back_populates="users")
    ratings = relationship("Rating", back_populates="user")

# Group model
class Group(Base):
    __tablename__ = "groups"
    group_code = Column(String, primary_key=True, index=True)
    family_name = Column(String, nullable=False)
    creator = Column(String, nullable=False)
    users = relationship("User", back_populates="group")
    dishes = relationship("Dish", back_populates="group")
    polls = relationship("Poll", back_populates="group")
    schedule_entries = relationship("ScheduleEntry", back_populates="group")

# Dish model
class Dish(Base):
    __tablename__ = "dishes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    category = Column(String, nullable=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    group = relationship("Group", back_populates="dishes")
    ratings = relationship("Rating", back_populates="dish")
    poll_options = relationship("PollOption", back_populates="dish")
    schedule_entries = relationship("ScheduleEntry", back_populates="dish")

# Rating model
class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    rating = Column(Float, nullable=False)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="ratings")
    dish = relationship("Dish", back_populates="ratings")

# Poll model
class Poll(Base):
    __tablename__ = "polls"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    group = relationship("Group", back_populates="polls")
    options = relationship("PollOption", back_populates="poll")

# PollOption model
class PollOption(Base):
    __tablename__ = "poll_options"
    id = Column(Integer, primary_key=True, index=True)
    poll_id = Column(Integer, ForeignKey("polls.id"))
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    option_text = Column(String, nullable=False)
    votes = Column(Integer, default=0)
    poll = relationship("Poll", back_populates="options")
    dish = relationship("Dish", back_populates="poll_options")

# ScheduleEntry model
class ScheduleEntry(Base):
    __tablename__ = "schedule"
    id = Column(Integer, primary_key=True, index=True)
    group_code = Column(String, ForeignKey("groups.group_code"), index=True)
    dish_id = Column(Integer, ForeignKey("dishes.id"))
    scheduled_date = Column(DateTime, nullable=False)
    group = relationship("Group", back_populates="schedule_entries")
    dish = relationship("Dish", back_populates="schedule_entries")
