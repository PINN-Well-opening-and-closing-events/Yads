import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from yads.predictors.Neural_Networks.create_cnn import create_cnn_1D

train = pd.read_csv(
    "data/train_first_approach.csv", converters={"S": literal_eval, "P": literal_eval}
)
test = pd.read_csv(
    "data/test_first_approach.csv", converters={"S": literal_eval, "P": literal_eval}
)

trainX, trainY = train[["q", "total_time"]].to_numpy(), np.array(list(train["S"]))
testX, testY = test[["q", "total_time"]].to_numpy(), np.array(list(test["S"]))


print(trainY.shape, trainX.shape)
scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

model = create_cnn_1D(2, 1, output_shape=256, filters=(4, 8, 16, 32, 64, 128, 256))

opt = Adam(learning_rate=1e-3, decay=1e-3 / 50)
model.compile(loss="mse", optimizer=opt)

history = model.fit(
    trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=64
)

print("Evaluate on test data")
results = model.evaluate(testX, testY, batch_size=64)
print("test loss, test acc:", results)
model.save("models/CNN")

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.yscale("log")
plt.title("CNN model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
