import pickle
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "size": 12})
rc("figure", **{"figsize": (5, 3)})

loss_dict = pickle.load(open("models/loss_dict_1500.pkl", "rb"))
print(
    np.argmin(loss_dict["val_loss"]),
    loss_dict["val_loss"][1368],
    loss_dict["train_loss"][1368],
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
plt.savefig(f"local_approach_2d_loss.pdf", bbox_inches="tight")
# plt.show()
