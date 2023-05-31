import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from yads.predictors.Neural_Networks.create_mlp import create_256_mlp

df = pd.read_csv(
    "data/getting_sirious.csv", converters={"S": literal_eval, "P": literal_eval}
)

X = df[["q", "total_time"]]
y = df[["S"]]


map_shape = (y.shape[0], len(df["S"].loc[0]))

(trainX, testX, trainY, testY) = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=False
)

trainAttrX, testAttrX = (
    trainX[["q", "total_time"]].to_numpy(),
    testX[["q", "total_time"]].to_numpy(),
)
trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))

scaler = StandardScaler()
scaler.fit(trainAttrX)
trainAttrX, testAttrX = scaler.transform(trainAttrX), scaler.transform(testAttrX)
model = create_256_mlp(2, 256)
print(model.summary())

opt = Adam(learning_rate=1e-3, decay=1e-3 / 150)
model.compile(loss="mse", optimizer=opt)

history = model.fit(
    trainAttrX, trainY, validation_data=(testAttrX, testY), epochs=150, batch_size=64
)

print("Evaluate on test data")
results = model.evaluate(testAttrX, testY, batch_size=128)
print("test loss, test acc:", results)

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.yscale("log")
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
