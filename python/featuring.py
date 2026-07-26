import pandas as pd 
import numpy as np 
import seaborn as sns

df = pd.read_csv("cleaned_data.csv")

cols = [
        'match_id',
        'innings',
        'batting_team',
        'bowling_team',
        'ball_no',
        'runs_total',
        'player_out',
        'runs_target',
        'match_won_by', 
        'curr_ball'
       ]

df = df[df["innings"]==2]

df["successfull_chase"] =(df["batting_team"]==
                          df["match_won_by"]).astype(int)

df["is_wicket"] = (df["player_out"]!="0").astype(int)

df["current_wicket"] = df.groupby("match_id")["is_wicket"].cumsum()

df["current_score"] = df.groupby("match_id")["runs_total"].cumsum()

df["current_rr"] = df["current_score"]*6/df["curr_ball"]

df["runs_per_wicket"] = df["current_score"]/df["current_wicket"]

df["required_runs"] = df["runs_target"] - df["current_score"]

df["balls_left"] = 120 - df["curr_ball"]

df["wickets_left"] = 10 - df["current_wicket"]

df["required_rr"] = df["required_runs"]*6/df["balls_left"]

df["required_runs_per_wicket"] = df["required_runs"]/df["wickets_left"]

df = df.rename(columns={"curr_ball":"current_balls",
                "runs_target":"target"
                })

df = df.drop(["player_out","innings","ball_no","runs_total","player_out","is_wicket","match_won_by"],axis=1)

df = df.replace([float("inf"),float("-inf")],0)

df = df.fillna(0)

for col in df.columns:
      if df[col].dtypes != "object":
            print(f"\n{col} : ")
            print(f"   max : {df[col].max()}")
            print(f"   min : {df[col].min()}")
            print(f"   avg : {df[col].mean()}")


df.to_csv("ml_ready.csv",index=False)


