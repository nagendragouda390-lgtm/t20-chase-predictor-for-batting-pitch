import pandas as pd
import numpy as np
import seaborn as sns

from funct import featuring

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

import joblib
# reading CSV file

import pandas as pd
import numpy as np
def model(df,name):

    df = pd.read_csv(df)

    df = df.copy()

    # Sort balls correctly
    df = df.sort_values(
        ["match_id", "ball_number"]
    ).reset_index(drop=True)

    groups = ["match_id"]

    # -----------------------------
    # Current score
    # -----------------------------
    df["current_score"] = (
        df.groupby(groups)["total_runs"]
        .cumsum()
    )

    # -----------------------------
    # Current wickets
    # -----------------------------
    df["current_wicket"] = (
        df.groupby(groups)["is_wicket"]
        .cumsum()
    )

    # Current balls
    df = df.rename(
        columns={"ball_number": "current_balls"}
    )

    # -----------------------------
    # Current run rate
    # -----------------------------
    df["current_rr"] = np.where(
        df["current_balls"] > 0,
        df["current_score"] * 6 / df["current_balls"],
        0
    )

    # -----------------------------
    # Runs per wicket
    # -----------------------------
    df["runs_per_wicket"] = np.where(
        df["current_wicket"] > 0,
        df["current_score"] / df["current_wicket"],
        df["current_score"]
    )

    # -----------------------------
    # Runs scored on each ball
    # -----------------------------
    df["ball_runs"] = (
        df.groupby(groups)["total_runs"]
        .transform(lambda x: x)
    )

    # -----------------------------
    # Wickets on each ball
    # -----------------------------
    df["ball_wickets"] = df["is_wicket"]

    # -----------------------------
    # Last 30 balls = last 5 overs
    # -----------------------------
    df["last_5_over_runs"] = (
        df.groupby(groups)["ball_runs"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=1
            ).sum()
        )
    )

    df["last_5_over_wickets"] = (
        df.groupby(groups)["ball_wickets"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=1
            ).sum()
        )
    )

    # -----------------------------
    # Required runs
    # -----------------------------
    df["required_runs"] = (
        df["target"] - df["current_score"]
    )

    # -----------------------------
    # Wickets left
    # -----------------------------
    df["wickets_left"] = (
        10 - df["current_wicket"]
    )

    # -----------------------------
    # Balls left
    # -----------------------------
    df["balls_left"] = (
        120 - df["current_balls"]
    )

    # Prevent negative values
    df["balls_left"] = df["balls_left"].clip(lower=0)

    # -----------------------------
    # Required run rate
    # -----------------------------
    df["required_rr"] = np.where(
        df["balls_left"] > 0,
        df["required_runs"] * 6 / df["balls_left"],
        0
    )

    # -----------------------------
    # Required runs per wicket
    # -----------------------------
    df["required_runs_per_wicket"] = np.where(
        df["wickets_left"] > 0,
        df["required_runs"] / df["wickets_left"],
        0
    )

    # -----------------------------
    # Clean data
    # -----------------------------
    df = df.replace(
        [np.inf, -np.inf],
        0
    )

    df = df.fillna(0)

    # -----------------------------
    # Select features
    # -----------------------------
    df = df[
        [
            "match_id",
            "batting_team",
            "bowling_team",
            "target",
            "current_score",
            "current_wicket",
            "current_balls",
            "current_rr",
            "runs_per_wicket",
            "last_5_over_runs",
            "last_5_over_wickets",
            "required_runs",
            "wickets_left",
            "balls_left",
            "required_rr",
            "required_runs_per_wicket",
            "batting_team_won"
        ]
    ]      

    X = df.drop(["batting_team_won","match_id"],axis=1) # features
    y = df["batting_team_won"] # target

# Category columns
    cat_col = [col for col in X.columns if X[col].dtypes == "str"]
# numerical columns
    num_col = [col for col in X.columns if col not in cat_col]

# creating preprocessing step for pipeline
    proc = ColumnTransformer([
         ("cat", OneHotEncoder(handle_unknown="ignore"),cat_col),
         ("num", QuantileTransformer(),num_col)
       ])
# creating pipeline
    pipe_lr = Pipeline([
            ("pre",proc),
            ("model", LogisticRegression())
          ])

# training pipeline
    pipe_lr.fit(X,y)

# dumping model     
    joblib.dump(pipe_lr,name)
    print("model_dumped_successfully!!")

model("200+data.csv","model_200_+.pkl")
model("175+data.csv","model_175_+.pkl")
model("150+data.csv","model_150_+.pkl")
model("120+data.csv","model_120_+.pkl")
model("low_data.csv","model_low_score.pkl")

