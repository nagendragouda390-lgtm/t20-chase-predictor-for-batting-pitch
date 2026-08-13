import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("ipl_ball_by_ball.csv",low_memory=False)

cols = [
        'match_id',
        'innings',
        'batting_team',
        'bowling_team',
        'ball_number',
        'total_runs',
        'is_wicket',
        'batting_team_won'
        ]

df = df[cols]

target_df = df[df["innings"]==1].groupby(["match_id"])["total_runs"].sum().reset_index(name="target-1")
target_df["target"] = target_df["target-1"] + 1
df = pd.merge(df,target_df[["match_id","target"]],how="inner",on="match_id")

df = df[df["innings"] == 2]

df_200 = df[df["target"]>=200]
df_175 = df[(df["target"]>=175) & (df["target"]<200)]
df_150 = df[(df["target"]>=150) & (df["target"]<175)]
df_120 = df[(df["target"]>=120) & (df["target"]<150)]
df_low = df[(df["target"]>=50) & (df["target"]<120)]

df_200.to_csv("200+data.csv",index=False)
df_175.to_csv("175+data.csv",index=False)
df_150.to_csv("150+data.csv",index=False)
df_120.to_csv("120+data.csv",index=False)
df_low.to_csv("low_data.csv",index=False)


