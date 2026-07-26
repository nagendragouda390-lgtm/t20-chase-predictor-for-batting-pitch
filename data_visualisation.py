import pandas as pd 
import numpy as np 
import seaborn as sns

df = pd.read_csv("ml_ready.csv")

cols = [
         'match_id',
         'batting_team', 
         'bowling_team', 
         'target', 
         'current_balls', 
         'successfull_chase', 
         'current_wicket', 
         'current_score', 
         'current_rr', 
         'runs_per_wicket', 
         'required_runs', 
         'balls_left', 
         'wickets_left', 
         'required_rr', 
         'required_runs_per_wicket'
       ]

import matplotlib.pyplot as plt

sns.scatterplot(x = df.index , y = df.target,hue=df.successfull_chase)

plt.savefig("target.png")
plt.show()

