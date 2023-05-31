import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from yads.predictors.Neural_Networks.create_multi_input_model import (
    create_multi_input_model,
)
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from yads.predictors.Neural_Networks.create_mlp import create_mlp
from yads.predictors.Neural_Networks.create_cnn import create_cnn_1D

df = pd.read_csv(
    "data/second_approach_5000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)
df["log_q"] = -np.log10(-df["q"])

X = df[["log_q", "dt", "S0"]]
y = df[["S"]]

map_shape = (y.shape[0], len(df["S0"].loc[0]))

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.8, random_state=42)

scaler = StandardScaler()
scaler.fit(trainX[["log_q", "dt"]])
trainX[["log_q", "dt"]], testX[["log_q", "dt"]] = (
    scaler.transform(trainX[["log_q", "dt"]]),
    scaler.transform(testX[["log_q", "dt"]]),
)

trainAttrX, testAttrX = (
    trainX[["log_q", "dt"]].to_numpy(),
    testX[["log_q", "dt"]].to_numpy(),
)
trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))

trainY_reshape = trainY.reshape((trainY.shape[0], 1, trainY.shape[1]))
testY_reshape = testY.reshape((testY.shape[0], 1, testY.shape[1]))


S0_train, S0_test = np.array(list(trainX["S0"])), np.array(list(testX["S0"]))

S0_train = S0_train.reshape((S0_train.shape[0], 1, S0_train.shape[1]))
S0_test = S0_test.reshape((S0_test.shape[0], 1, S0_test.shape[1]))

model_cnn = create_cnn_1D(1, map_shape[1], 16, filters=(128, 64, 32))
model_mlp = create_mlp(2, 32)

concatenated = concatenate([model_mlp.output, model_cnn.output])
model = create_multi_input_model(model_cnn, model_mlp, filters=(128, 256))

opt = Adam(learning_rate=1e-3, decay=1e-3 / 200)
model.compile(loss="mse", optimizer=opt)


print("MLP:", model_mlp.input.shape, trainAttrX.shape)
print("CNN:", model_cnn.input.shape, S0_train.shape)
print("MERGED:", model.output.shape, trainY_reshape.shape)

history = model.fit(
    [trainAttrX, S0_train],
    trainY_reshape,
    validation_data=([testAttrX, S0_test], testY_reshape),
    epochs=200,
    batch_size=64,
)

print("Evaluate on test data")
results = model.evaluate([testAttrX, S0_test], testY, batch_size=64)
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
