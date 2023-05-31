import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error
from pickle import dump

train = pd.read_csv(
    "data/train_1000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)
test = pd.read_csv(
    "data/test_4000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)
train["log_q"] = -np.log10(-train["q"])
test["log_q"] = -np.log10(-test["q"])

train["S0_single"] = [elt[0] for elt in train["S0"]]
test["S0_single"] = [elt[0] for elt in test["S0"]]

trainX, trainY = (
    train[["log_q", "total_time", "S0_single"]].to_numpy(),
    np.array(list(train["S"])),
)
testX, testY = (
    test[["log_q", "total_time", "S0_single"]].to_numpy(),
    np.array(list(test["S"])),
)


print(trainY.shape, trainX.shape)
scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

model = RadiusNeighborsRegressor(radius=1.0, weights="distance")
model.fit(trainX, trainY)

print("train mse: ", mean_squared_error(trainY, model.predict(trainX), squared=False))
print("test mse: ", mean_squared_error(testY, model.predict(testX), squared=False))
# save the model
dump(model, open("models/model_knn_dist.pkl", "wb"))
# save the scaler
dump(scaler, open("models/scaler.pkl", "wb"))
