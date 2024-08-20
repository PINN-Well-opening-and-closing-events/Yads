import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc

loss_dict = pickle.load(open("models/loss_dict_1500.pkl", "rb"))
print(
    np.argmin(loss_dict["val_loss"]),
    loss_dict["val_loss"][1368],
    loss_dict["train_loss"][1368],
)
x = list(range(1, len(loss_dict["train_loss"]) + 1, 10))
fig = plt.plot(figsize=(16, 16))
plt.plot(x, loss_dict["train_loss"][::10], label="Train loss")
plt.plot(x, loss_dict["val_loss"][::10], label="Test loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test loss")
plt.grid(True)
plt.legend()
plt.savefig(f"local_approach_2d_loss.pdf", bbox_inches="tight")
# plt.show()
