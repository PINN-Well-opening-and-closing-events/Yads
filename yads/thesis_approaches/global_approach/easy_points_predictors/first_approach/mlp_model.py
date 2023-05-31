import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from yads.predictors.Neural_Networks.create_mlp import create_256_mlp

train = pd.read_csv(
    "data/train_1000.csv", converters={"S": literal_eval, "P": literal_eval}
)
test = pd.read_csv(
    "data/test_4000.csv", converters={"S": literal_eval, "P": literal_eval}
)

train["log_q"] = -np.log10(-train["q"])
test["log_q"] = -np.log10(-test["q"])

trainX, trainY = train[["log_q", "total_time"]].to_numpy(), np.array(list(train["S"]))
testX, testY = test[["log_q", "total_time"]].to_numpy(), np.array(list(test["S"]))

print(trainY.shape, trainX.shape)
scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

model = create_256_mlp(2, 256)

opt = Adam(learning_rate=1e-3, decay=1e-3 / 400)
model.compile(loss="mse", optimizer=opt)

history = model.fit(
    trainX, trainY, validation_data=(testX, testY), epochs=400, batch_size=64
)

print("Evaluate on test data")
results = model.evaluate(testX, testY, batch_size=64)
print("test loss, test acc:", results)
model.save("models/MLP")

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.yscale("log")
plt.title("MLP model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
