"""
Trains model_lr.pkl using the SAME pipeline architecture as the original
training script (ColumnTransformer -> OrdinalEncoder + StandardScaler ->
LogisticRegression), since the original ml_ready.csv / functions.py were
not available. Synthetic but cricket-realistic chase data is generated
so the model behaves sensibly for the web app.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

RNG = np.random.default_rng(42)

TEAMS = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants",
]

N = 25000

rows = []
for _ in range(N):
    batting_team, bowling_team = RNG.choice(TEAMS, size=2, replace=False)

    target = int(RNG.normal(172, 28))
    target = max(90, min(target, 260))

    current_balls = int(RNG.integers(6, 120))         # balls bowled so far
    balls_left = 120 - current_balls

    # plausible current run rate around the innings
    base_rr = RNG.normal(8.2, 2.0)
    base_rr = max(2.0, base_rr)
    current_score = int(round(base_rr * (current_balls / 6)))
    current_score = min(current_score, target - 1) if target - 1 > 0 else current_score
    current_score = max(current_score, 0)

    current_wicket = int(RNG.integers(0, 10))
    wickets_left = 10 - current_wicket

    current_rr = round((current_score / (current_balls / 6)), 2) if current_balls > 0 else 0.0
    runs_per_wicket = round(current_score / (current_wicket + 1), 2)

    # last 5 overs (30 balls) window
    window_balls = min(30, current_balls)
    last_5_over_runs = int(round(current_rr * (window_balls / 6))) if current_balls > 0 else 0
    last_5_over_wickets = int(RNG.integers(0, min(5, current_wicket) + 1))

    required_runs = max(target - current_score, 0)
    required_rr = round((required_runs / (balls_left / 6)), 2) if balls_left > 0 else 99.0
    required_runs_per_wicket = round(required_runs / (wickets_left + 1), 2)

    rows.append([
        batting_team, bowling_team, target, current_score, current_wicket,
        current_balls, current_rr, runs_per_wicket, last_5_over_runs,
        last_5_over_wickets, required_runs, wickets_left, balls_left,
        required_rr, required_runs_per_wicket
    ])

cols = [
    "batting_team", "bowling_team", "target", "current_score", "current_wicket",
    "current_balls", "current_rr", "runs_per_wicket", "last_5_over_runs",
    "last_5_over_wickets", "required_runs", "wickets_left", "balls_left",
    "required_rr", "required_runs_per_wicket"
]

df = pd.DataFrame(rows, columns=cols)

# ---- realistic probabilistic label for successful chase ----
# Higher chance of winning when required run rate is close to / below
# current run rate, more wickets and balls remain, and required_rr isn't huge.
z = (
    -0.55 * (df["required_rr"] - df["current_rr"])
    + 0.28 * df["wickets_left"]
    + 0.010 * df["balls_left"]
    - 0.05 * df["required_runs_per_wicket"]
    + RNG.normal(0, 1.4, size=len(df))   # noise -> cricket is unpredictable
)
prob = 1 / (1 + np.exp(-z / 3.0))
df["successfull_chase"] = (RNG.random(len(df)) < prob).astype(int)

X = df[cols]
y = df["successfull_chase"]

cat_col = ["batting_team", "bowling_team"]
num_col = [c for c in X.columns if c not in cat_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_col),
    ("num", StandardScaler(), num_col),
])

pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

pipe_lr.fit(X_train, y_train)

y_pred = pipe_lr.predict(X_test)
y_proba = pipe_lr.predict_proba(X_test)[:, 1]

accu = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
conf = confusion_matrix(y_test, y_pred)

print(f"accuracy  : {accu:.3f}")
print(f"recall    : {recall:.3f}")
print(f"precision : {precision:.3f}")
print(f"roc_auc   : {auc:.3f}")
print(f"confusion :\n{conf}")

# fit on full data for the deployed model (matches original script's approach)
pipeline_final = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_col),
        ("num", StandardScaler(), num_col),
    ])),
    ("model", LogisticRegression(max_iter=1000)),
])
pipeline_final.fit(X, y)

joblib.dump(pipeline_final, "model/model_lr.pkl")
joblib.dump(TEAMS, "model/teams.pkl")
print("model dumped successfully -> model/model_lr.pkl")
