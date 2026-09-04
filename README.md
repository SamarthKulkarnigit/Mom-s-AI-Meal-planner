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

```mermaid
flowchart TD
    U[Family member] --> S[Streamlit frontend]
    S -->|JWT: auth, stats, plan generation, replacement| A[FastAPI backend]
    S -->|JWT revalidated via /me before trusted compatibility reads/writes| D[SQLAlchemy compatibility layer]
    A --> R[Hybrid recommendation engine]
    R --> C[Grounded candidate set]
    C --> G[Gemini planning and explanations]
    G --> V[Backend validation]
    V --> P[(PostgreSQL / SQLite)]
    A --> P
    D --> P
    V -. failure .-> F[Deterministic fallback]
    F --> P
```

### Tech Stack
- **Python 3**
- **Frontend**: Streamlit, Pandas, Plotly
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, Pydantic
- **Machine Learning**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Persistence**: Unified SQLAlchemy models backed by SQLite (local) or PostgreSQL (production)

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
You can configure the application database connection via the `DATABASE_URL` environment variable. By default, it will use a local SQLite file.

For local development (default):
```bash
export DATABASE_URL="sqlite:///./backend/data/mealplanner.db"
```

For production (PostgreSQL):
```bash
export DATABASE_URL="postgresql://user:password@host:port/dbname"
```
Copy `.env.example` and set the values in your shell or deployment secret manager.
Never commit real credentials. When `ENVIRONMENT=production`, `SECRET_KEY` is
required and startup fails safely if it is absent.

### 4. Data Migration (Important for Phase 1)
If you have legacy CSV data in the `data/` folder, migrate it to the new SQL database using the migration utility:
```bash
python scripts/migrate_csv_to_sql.py
```
This script safely parses all CSV files and inserts them into the SQLAlchemy database. It is idempotent and safe to run multiple times.

### 5. Start the Backend API (Terminal 1)
```bash
uvicorn backend.main:app --reload
```
The backend will automatically initialize the database and run on `http://127.0.0.1:8000`.

### 6. Start the Frontend Application (Terminal 2)
Open a new terminal, activate the virtual environment, and run:
```bash
PYTHONPATH=. streamlit run main.py
```
The application will be accessible at `http://localhost:8501`.

## Production deployment

Use one Aiven PostgreSQL database for both services during this shipping phase.
The Streamlit process still contains a server-side SQLAlchemy compatibility
layer, so its `DATABASE_URL` must exactly match Render's `DATABASE_URL`.

### Render backend

1. Create a Python web service from this repository.
2. Set the start command:
   `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Set `ENVIRONMENT=production`, `DATABASE_URL`, `SECRET_KEY`,
   `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.6-flash`, and
   `GEMINI_TIMEOUT_MS=60000` as secret environment variables.
4. Verify `https://<render-service>/health` returns
   `{"status":"ok","database":"ok"}`.

### Streamlit Community Cloud frontend

1. Deploy `main.py` from this repository.
2. Add `API_URL=https://<render-service>` and the same Aiven `DATABASE_URL`
   using Streamlit secrets/environment configuration.
3. Start command: `streamlit run main.py` (Community Cloud invokes this for
   the selected entrypoint).

### Submission smoke test

Create a family, generate a seven-day plan, replace one day, submit feedback,
open analytics, regenerate, then temporarily remove `GEMINI_API_KEY` from a
non-production test instance and confirm the deterministic fallback succeeds.

## Design Decisions
- **Unified Persistence Layer**: All data (Users, Groups, Dishes, Ratings, Polls, Schedules) has been migrated from CSVs to a cloud-ready SQL database using SQLAlchemy. `db.py` acts as a compatibility wrapper that directly queries the database and returns Pandas DataFrames, preventing unnecessary rewrites of the ML algorithms or UI components.
- **API Layer**: Authentication and schedule mutations use FastAPI. Remaining
  legacy UI data access stays server-side through `db.py` and is gated by JWT
  revalidation on every authenticated Streamlit rerun.
- **A-Res Sampling**: Chosen for the recommendation engine to provide a mathematically sound way of randomizing selections while still strongly favoring highly-rated dishes.

## Post-submission maintenance

- Move the remaining server-side `db.py` compatibility calls behind FastAPI.
- Replace additive startup migrations with a versioned migration tool.
- Preserve immutable rating events for more rigorous longitudinal analytics.

## Project Status
This is a personal/student project created by Samarth Kulkarni.
