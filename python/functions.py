from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,recall_score, precision_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def split(df,X,y):
        gss = GroupShuffleSplit(test_size=0.2,random_state = 43)
        train_idx,test_idx = next(gss.split(X,y,groups=df["match_id"]))
        X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
        X_test ,y_test  = X.iloc[test_idx],y.iloc[test_idx]
        return X_train,X_test,y_train,y_test


def pipeline(model,encoder,scaler,cat,num):
      preprocessor = ColumnTransformer([
              ("cat", encoder,cat),
              ("num", scaler,num)
              ])

      pipe = Pipeline([
          ("preprocessor", preprocessor),
          ("model",model)
          ])
      return pipe


def training(model,X_train,y_train):
      model.fit(X_train,y_train)
      return model


def testing(model,X_test,y_test):
      y_pred = model.predict(X_test)
      
      accuracy = accuracy_score(y_test,y_pred)
      recall = recall_score(y_test,y_pred)
      precision = precision_score(y_test,y_pred)
      conf = confusion_matrix(y_test,y_pred)
      
      return y_pred, accuracy,recall, precision, conf



      
      






