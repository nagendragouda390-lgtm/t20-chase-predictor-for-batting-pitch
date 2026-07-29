import pandas as pd 
import numpy as np 
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,recall_score, precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
pipe_lr = pipeline(LogisticRegression(),OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),StandardScaler(),cat_col,num_col)

pipe_lr = training(pipe_lr,X_train,y_train)

y_pred, accu, recall, precision, conf = testing(pipe_lr,X_test,y_test)

"""
LR + OE
pred      : [1 1 1 ... 1 1 1]
accuracy  : 0.753
recall    : 0.683
precision : 0.824
conf      : [[1432  287]
             [ 621 1341]]

DTC + OE
pred      : [1 1 1 ... 1 1 1]
accuracy  : 0.699
recall    : 0.667
precision : 0.742
conf      : [[1264  455]
             [ 654 1308]]

RFC + OE
pred      : [1 1 1 ... 1 1 1]
accuracy  : 0.747
recall.   : 0.648
precision : 0.841
conf      : [[1478  241]
             [ 690 1272]]



"""
pipe_dtc = pipeline(DecisionTreeClassifier(random_state=43),OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),StandardScaler(),cat_col,num_col)

pipe_dtc = training(pipe_dtc,X_train,y_train)

y_dtc, accu_dtc, recall_dtc, precision_dtc, conf_dtc = testing(pipe_dtc,X_test,y_test)

#print(f"pred : {y_dtc}\naccuracy : {accu_dtc}\nrecall : {recall_dtc}\nprecision : {precision_dtc}\nconf : {conf_dtc}")

pipe_rfc = pipeline(RandomForestClassifier(random_state=54),OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),StandardScaler(),cat_col,num_col)

pipe_rfc = training(pipe_rfc,X_train,y_train)

y_rfc, accu_rfc, recall_rfc, precision_rfc, conf_rfc = testing(pipe_rfc,X_test,y_test)

#print(f"pred : {y_rfc}\naccuracy : {accu_rfc}\nrecall : {recall_rfc}\nprecision : {precision_rfc}\nconf : {conf_rfc}")

from sklearn.model_selection import cross_val_score

score = cross_val_score(pipe_lr,X_test,y_test,cv=5)

score = pd.DataFrame(score)
print(score.mean())

"""
croos validation score of lr

1  0.799186
2  0.464674
3  0.922554
4  0.817935
5  0.748641

mean = 0.750

which is good it is almost equal to testing accuracy.
cricket is unpredictable game.

"""
import joblib

joblib.dump(pipe_lr,"model_lr.pkl")

print("model dumped successfuly !!")

