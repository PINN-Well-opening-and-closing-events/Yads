import pickle
import matplotlib.pyplot as plt
import numpy as np

loss_dict = pickle.load(open("model/loss_dict_7300.pkl", "rb"))
print(
    np.argmin(loss_dict["val_loss"]),
    loss_dict["val_loss"][7295],
    loss_dict["train_loss"][7295],
)
x = list(range(1, len(loss_dict["train_loss"]) + 1))
fig = plt.plot(figsize=(16, 16))
plt.plot(x, loss_dict["train_loss"], label="Train loss")
plt.plot(x, loss_dict["val_loss"], label="Test loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test loss")
plt.legend()
# plt.show()
