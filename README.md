# Mom's AI Meal Planner

## Overview
Mom's AI Meal Planner is a smart, adaptive weekly meal recommendation system designed to help families plan their meals effortlessly. 
By incorporating family preferences, suggestions, polls, and historical ratings, the application dynamically generates a weekly schedule of dishes. It uses content similarity and collaborative filtering techniques to avoid fatigue while ensuring everyone's favorite meals are served.

## Features
- **Family Management**: Create or join family groups securely.
- **Meal Suggestions & Polling**: Family members can suggest new dishes and vote on them to add to the rotation.
- **Meal Ratings**: Members rate daily meals, which the recommendation engine uses to learn preferences over time.
- **Meal Scheduler**: Automatically generate a weekly meal plan tailored to the family's tastes.
- **Analytics & Daily Feedback**: Track the top-rated meals, total dishes in rotation, and provide comments on meals.
- **Smart Recommendations**: Hybrid recommendation engine using time-decayed rating averages, content similarity, and collaborative filtering.

## Technical Architecture

The application is modularized to separate the frontend UI from the backend business logic and data persistence layer.

```
[ Frontend (Streamlit) ]
         |
    (api_client.py)
         |
         v
[ Backend API (FastAPI) ]
         |
         v
[ Data Layer (SQLite & CSVs) ]
```

### Tech Stack
- **Python 3**
- **Frontend**: Streamlit, Pandas, Plotly
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Pydantic
- **Machine Learning**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Persistence**: SQLite (for User/Group auth) + CSV (for ratings, polls, and schedules)

## Recommendation System

The recommendation engine (`ml_recommender.py`) uses a hybrid scoring approach to build the weekly meal plan:
1. **Time-Decayed Average Ratings**: Ratings from recent weeks are weighted more heavily than older ratings to capture changing tastes.
2. **Content Similarity**: TF-IDF and cosine similarity ensure the generated plan doesn't include dishes that are too similar to each other in the same week.
3. **Collaborative Filtering**: Evaluates similarity matrices of user ratings to find popular dish patterns.
4. **Hybrid Score**: A combination of average rating score, popularity (votes), collaborative filtering score, and content score.
5. **Randomization (A-Res)**: Uses Adjusted Reservoir Sampling with probabilities derived from the hybrid scores to inject variety and avoid repeating the exact same plan every week.

## Setup Instructions

This project uses two terminals to run the backend API and the frontend application simultaneously.

### 1. Clone and Environment
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/SamarthKulkarnigit/Mom-s-AI-Meal-planner.git
cd Mom-s-AI-Meal-planner
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the example environment file and update it if necessary:
```bash
cp .env.example .env
```
Ensure `SECRET_KEY` is set in the `.env` file for secure session management.

### 4. Start the Backend API (Terminal 1)
```bash
uvicorn backend.main:app --reload
```
The backend will automatically initialize the `mealplanner.db` database and run on `http://127.0.0.1:8000`.

### 5. Start the Frontend Application (Terminal 2)
Open a new terminal, activate the virtual environment, and run:
```bash
streamlit run main.py
```
The application will be accessible at `http://localhost:8501`.

## Screenshots
> *Placeholder for application screenshots (e.g., Home Dashboard, Scheduler, Analytics).*

## Design Decisions
- **API Layer**: An API layer was introduced so the Streamlit frontend remains stateless and can easily be replaced or supplemented by mobile clients in the future.
- **Hybrid Persistence**: SQLite handles secure relational data (Users, Groups, Passwords) via SQLAlchemy, while Pandas/CSV handles time-series and log data (ratings, polls) for rapid prototyping and easy data inspection.
- **A-Res Sampling**: Chosen for the recommendation engine to provide a mathematically sound way of randomizing selections while still strongly favoring highly-rated dishes.

## Future Improvements
- **Production Database**: Migrate all CSV data to PostgreSQL/SQLite for better concurrency and integrity.
- **Docker/Containerization**: Containerize the frontend and backend using `docker-compose` for easier deployment.
- **Deployment**: Deploy on cloud providers (e.g., AWS, Render, Heroku).
- **Nutrition-Aware Recommendations**: Integrate a nutritional API to balance macros in the weekly schedule.
- **Grocery List Generation**: Automatically generate grocery lists based on the scheduled meals.

## Project Status
This is a personal/student project created by Samarth Kulkarni.
