import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

import pickle

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

train["log_opt_dt"] = np.log10(train["optimal_dt"])
test["log_opt_dt"] = np.log10(test["optimal_dt"])

train = train[train["case"] == "giga_hard"]
test = test[test["case"] == "giga_hard"]

trainX, trainY = train[["log_q", "dt_init"]].values, train["log_opt_dt"]
testX, testY = test[["log_q", "dt_init"]].values, test["log_opt_dt"]

scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

model = XGBRegressor()
model.fit(trainX, trainY)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(
    model, trainX, trainY, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
)
# force scores to be positive
scores = np.absolute(scores)
print("Train mean MAE: %.3f (%.3f)" % (scores.mean(), scores.std()))

print(f"test MAE: {mean_squared_error(testY, model.predict(testX))}")
# save the model
pickle.dump(model, open("models/dt_predict_model_xgboost_110_100_P.pkl", "wb"))
# save the scaler
pickle.dump(scaler, open("models/dt_predict_scaler_xgboost_110_100_P.pkl", "wb"))
