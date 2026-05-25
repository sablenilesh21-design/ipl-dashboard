# 🏏 IPL Analytics Dashboard

## Setup & Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place data files** — ensure these two files are in the same folder as `app.py`:
   - `matches.csv`
   - `deliveries.csv`

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

## Features
- **Batting Tab** — Top run scorers, SR vs Average scatter, 4s & 6s, Orange Cap
- **Bowling Tab** — Top wicket takers, Economy distribution, Purple Cap
- **Teams Tab** — Win %, Toss impact, Season trends
- **Venues Tab** — Most used venues, Chase vs Defend, Over-by-over scoring
- **Win Predictor** — Random Forest / Logistic Regression / Decision Tree with live probability bars

## Sidebar Filters
- Filter by **Season** (2007/08 → 2024) or **Team** — all charts update live
