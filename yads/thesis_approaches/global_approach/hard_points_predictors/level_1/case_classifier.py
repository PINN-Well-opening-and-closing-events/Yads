import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from pickle import dump
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics

train = pd.read_csv(
    "data/5000_10_100_newtons_110_100_P_train.csv",
    converters={"S": literal_eval, "P": literal_eval},
)
test = pd.read_csv(
    "data/5000_10_100_newtons_110_100_P_test.csv",
    converters={"S": literal_eval, "P": literal_eval},
)

train["log_q"] = np.log10(-train["q"])
test["log_q"] = np.log10(-test["q"])


map_dict = {"gentle": 0, "hard": 0, "giga_hard": 1}


train["class"] = train["case"].map(map_dict)
test["class"] = test["case"].map(map_dict)

trainX, trainY = train[["log_q", "dt_init"]].values, train["class"]
testX, testY = test[["log_q", "dt_init"]].values, test["class"]

scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)


model = XGBClassifier(eval_metric="mlogloss")
model.fit(trainX, trainY)

trainX_pred = model.predict(trainX)
testX_pred = model.predict(testX)

print("TRAIN RESULTS")
print(metrics.classification_report(trainX_pred, trainY))
print(metrics.confusion_matrix(trainX_pred, trainY))

print("TEST RESULTS")
print(metrics.classification_report(testX_pred, testY))
print(metrics.confusion_matrix(testX_pred, testY))

# save the model
dump(model, open("models/case_classifier_model_xgboost_110_100_P.pkl", "wb"))
# save the scaler
dump(scaler, open("models/case_classifier_scaler_xgboost_110_100_P.pkl", "wb"))
