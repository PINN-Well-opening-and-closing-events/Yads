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

trainX = train[["S0"]]
trainY = train[["S"]]

testX = test[["S0"]]
testY = test[["S"]]

map_shape = (trainY.shape[0], len(trainX["S0"].loc[0]))

trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))
trainX, testX = np.array(list(trainX["S0"])), np.array(list(testX["S0"]))

print(trainY.shape, trainX.shape)

model = RadiusNeighborsRegressor(radius=1.0, weights="distance")
model.fit(trainX, trainY)

print("train mse: ", mean_squared_error(trainY, model.predict(trainX), squared=False))
print("test mse: ", mean_squared_error(testY, model.predict(testX), squared=False))
# save the model
dump(model, open("models/model_knn_dist.pkl", "wb"))
