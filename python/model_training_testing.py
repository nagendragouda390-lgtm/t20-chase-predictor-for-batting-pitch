import pandas as pd 
import numpy as np 
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,recall_score, precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, QuantileTransformer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from functions import * 

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
        'required_runs_per_wicket',
        'last_5_over_runs',
        'last_5_over_wickets'
        ]

head = df.head()

null = df.isna().sum()

df = df.drop_duplicates()

dup = df.duplicated().sum()


X = df[[
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
      "required_runs_per_wicket"
      ]]

y = df["successfull_chase"]

cat_col = []
num_col = []
for col in X.columns:
      if df[col].dtypes == "object":
            cat_col.append(col)
      else:
            num_col.append(col)
            
X_train,X_test,y_train,y_test =split(df,X,y)

"""
Shape :
  X_train : (15071, 15)
  X_test  : (3681, 15)
  y_train : (15071,)
  y_test  : (3681,)

"""
preprocessor = ColumnTransformer([
              ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),cat_col),
              ("num", QuantileTransformer(),num_col)
              ])

pipeline = Pipeline([
          ("preprocessor", preprocessor),
          ("model", LogisticRegression(max_iter=1000))
          ])
pipeline.fit(X,y)

import joblib

joblib.dump(pipeline,"model_lr.pkl")

print("model dumped successfuly !!")

