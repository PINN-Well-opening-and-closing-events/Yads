import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from ast import literal_eval
from yads.predictors.easy_points_predictors.level_0.Unet import little_UNET

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

trainY_reshape = trainY.reshape((trainY.shape[0], 1, trainY.shape[1]))
testY_reshape = testY.reshape((testY.shape[0], 1, testY.shape[1]))

S0_train, S0_test = np.array(list(trainX["S0"])), np.array(list(testX["S0"]))

# S0_train = S0_train.reshape((S0_train.shape[0], 1, S0_train.shape[1]))
# S0_test = S0_test.reshape((S0_test.shape[0], 1, S0_test.shape[1]))


# model = Unet()
model = little_UNET()
# model = create_cnn_1D(256, 1, 256, filters=(128, 128, 128))
opt = Adam(learning_rate=1e-3, decay=1e-3 / 200)
model.compile(loss="mse", optimizer=opt)

history = model.fit(
    S0_train,
    trainY_reshape,
    validation_data=(S0_test, testY_reshape),
    epochs=200,
    batch_size=64,
)

model.save("models/UNET")

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.yscale("log")
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
