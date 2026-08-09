import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, g

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "data.db")
MODEL_175 = os.path.join(APP_DIR, "model", "model_175_+.pkl")
MODEL_200 = os.path.join(APP_DIR, "model","model_200_+.pkl")
MODEL_150 = os.path.join(APP_DIR, "model","model_150_+.pkl")
MODEL_120 = os.path.join(APP_DIR, "model","model_120_+.pkl")
MODEL_LOW = os.path.join(APP_DIR, "model","model_low_score.pkl")
TEAMS_PATH = os.path.join(APP_DIR, "model", "teams.pkl")

app = Flask(__name__)

model_low = joblib.load(MODEL_LOW)
model_120 = joblib.load(MODEL_120)
model_200 = joblib.load(MODEL_200)
model_175 = joblib.load(MODEL_175)
model_150 = joblib.load(MODEL_150)
TEAMS = sorted(joblib.load(TEAMS_PATH))


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visited_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            batting_team TEXT NOT NULL,
            bowling_team TEXT NOT NULL,
            target INTEGER NOT NULL,
            current_score INTEGER NOT NULL,
            current_wicket INTEGER NOT NULL,
            current_balls INTEGER NOT NULL,
            win_probability REAL NOT NULL,
            lose_probability REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            prediction_id INTEGER,
            stars INTEGER NOT NULL,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------- helpers
def record_visit():
    db = get_db()
    db.execute("INSERT INTO visits (visited_at) VALUES (?)", (datetime.utcnow().isoformat(),))
    db.commit()


def get_stats():
    db = get_db()
    total_visits = db.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"]
    total_predictions = db.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    avg_rating_row = db.execute("SELECT AVG(stars) AS a, COUNT(*) AS c FROM ratings").fetchone()
    avg_rating = round(avg_rating_row["a"], 1) if avg_rating_row["a"] else 0
    total_ratings = avg_rating_row["c"]
    return {
        "total_visits": total_visits,
        "total_predictions": total_predictions,
        "avg_rating": avg_rating,
        "total_ratings": total_ratings,
    }


def overs_to_balls(overs_str):
    """Convert cricket overs notation like '12.3' (12 overs, 3 balls) to total balls bowled."""
    overs_str = str(overs_str).strip()
    if "." in overs_str:
        whole, part = overs_str.split(".")
        whole = int(whole) if whole else 0
        balls_in_over = int(part[0]) if part else 0
        balls_in_over = min(balls_in_over, 5)
    else:
        whole = int(overs_str) if overs_str else 0
        balls_in_over = 0
    return whole * 6 + balls_in_over


# ---------------------------------------------------------------- routes
@app.route("/", methods=["GET"])
def index():
    record_visit()
    stats = get_stats()
    return render_template("index.html", teams=TEAMS, stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]
        target = int(request.form["target"])
        current_score = int(request.form["current_score"])
        current_wicket = int(request.form["current_wicket"])
        overs = request.form["overs"]
        last_5_over_runs = int(request.form.get("last_5_over_runs") or 0)
        last_5_over_wickets = int(request.form.get("last_5_over_wickets") or 0)

        if batting_team == bowling_team:
            raise ValueError("Batting and bowling team must be different.")

        current_balls = overs_to_balls(overs)
        current_balls = max(0, min(current_balls, 119))
        balls_left = max(120 - current_balls, 0)
        wickets_left = max(10 - current_wicket, 0)

        current_rr = round((current_score / (current_balls / 6)), 2) if current_balls > 0 else 0.0
        runs_per_wicket = round(current_score / (current_wicket + 1), 2)

        required_runs = max(target - current_score, 0)
        required_rr = round((required_runs / (balls_left / 6)), 2) if balls_left > 0 else 99.0
        required_runs_per_wicket = round(required_runs / (wickets_left + 1), 2)

        row = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "target": target,
            "current_score": current_score,
            "current_wicket": current_wicket,
            "current_balls": current_balls,
            "current_rr": current_rr,
            "runs_per_wicket": runs_per_wicket,
            "last_5_over_runs": last_5_over_runs,
            "last_5_over_wickets": last_5_over_wickets,
            "required_runs": required_runs,
            "wickets_left": wickets_left,
            "balls_left": balls_left,
            "required_rr": required_rr,
            "required_runs_per_wicket": required_runs_per_wicket,
        }])

        if target >= 200:
            select_model = model_200
        elif target >= 175:
            select_model = model_175
        elif target >= 150:
            select_model = model_150
        elif target >= 120:
            select_model = model_120
        else:
            select_model = model_low

        proba = select_model.predict_proba(row)[0]
        lose_probability = round(float(proba[0]) * 100, 1)
        win_probability = round(float(proba[1]) * 100, 1)

        db = get_db()
        cur = db.execute(
            """INSERT INTO predictions
               (created_at, batting_team, bowling_team, target, current_score,
                current_wicket, current_balls, win_probability, lose_probability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), batting_team, bowling_team, target,
             current_score, current_wicket, current_balls, win_probability, lose_probability),
        )
        db.commit()
        prediction_id = cur.lastrowid

        result = {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "target": target,
            "current_score": current_score,
            "current_wicket": current_wicket,
            "overs_display": f"{current_balls // 6}.{current_balls % 6}",
            "required_runs": required_runs,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "required_rr": required_rr,
            "current_rr": current_rr,
            "win_probability": win_probability,
            "lose_probability": lose_probability,
            "prediction_id": prediction_id,
        }
        return render_template("result.html", r=result)

    except (ValueError, KeyError) as exc:
        stats = get_stats()
        return render_template("index.html", teams=TEAMS, stats=stats, error=str(exc)), 400


@app.route("/rate", methods=["POST"])
def rate():
    stars = int(request.form["stars"])
    comment = request.form.get("comment", "").strip()
    prediction_id = request.form.get("prediction_id")
    db = get_db()
    db.execute(
        "INSERT INTO ratings (created_at, prediction_id, stars, comment) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), prediction_id or None, stars, comment),
    )
    db.commit()
    return redirect(url_for("thanks"))


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    stats = get_stats()
    return render_template("history.html", rows=rows, stats=stats)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
