import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error
from pickle import dump
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

train = pd.read_csv(
    "data/1000_10_100_newtons_2D_global_train.csv",
    converters={"S": literal_eval, "P": literal_eval},
)
test = pd.read_csv(
    "data/1000_10_100_newtons_2D_global_test.csv",
    converters={"S": literal_eval, "P": literal_eval},
)

train = train[train["case"] != "giga_hard"]
test = test[test["case"] != "giga_hard"]

train["log_q"] = np.log10(-train["q"])
test["log_q"] = np.log10(-test["q"])

trainX, trainY = train[["log_q", "dt_init"]].to_numpy(), np.array(list(train["S"]))
testX, testY = test[["log_q", "dt_init"]].to_numpy(), np.array(list(test["S"]))


print(trainY.shape, trainX.shape)
scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

model = MultiOutputRegressor(XGBRegressor())
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
dump(model, open("models/S_predict_model_hard_xgboost.pkl", "wb"))
# save the scaler
dump(scaler, open("models/S_predict_scaler_xgboost.pkl", "wb"))
