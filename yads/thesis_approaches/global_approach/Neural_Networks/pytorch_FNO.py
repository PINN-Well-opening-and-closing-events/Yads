"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
from ast import literal_eval
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import yads.mesh as ym

# from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


# Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack(
        [
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
        ],
        dim=-1,
    )


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1, 2, device=x.device
        )
        out_ft[:, :, : self.modes1] = compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.irfft(
            out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1),)
        )
        return x


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        # self.fc0 = nn.Linear(1, self.width)  # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(1, self.width)  # input channel is 1: a(x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # print()
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


################################################################
#  Loss
################################################################


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


################################################################
#  configurations
################################################################


sub = 2**3  # subsampling rate
h = 2**13 // sub  # total grid size divided by the subsampling rate
s = h

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 16
width = 64

################################################################
# read data
################################################################

train = pd.read_csv(
    "../easy_points_predictors/level_0/global_approach/data/train_1000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)
test = pd.read_csv(
    "../easy_points_predictors/level_0/global_approach/data/test_4000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)

trainX = train[["S0"]]
trainY = train[["S"]]

testX = test[["S0"]]
testY = test[["S"]]

map_shape = (trainY.shape[0], len(trainX["S0"].loc[0]))

trainY, testY = np.array(list(trainY["S"])), np.array(list(testY["S"]))

# trainY_reshape = trainY.reshape((trainY.shape[0], 1, trainY.shape[1]))
# testY_reshape = testY.reshape((testY.shape[0], 1, testY.shape[1]))

S0_train, S0_test = np.array(list(trainX["S0"])), np.array(list(testX["S0"]))

print(S0_train.shape, S0_test.shape)
print(trainY.shape, testY.shape)
# # Data is of the shape (number of samples, grid size)

# # cat the locations information

grid = ym.two_D.create_2d_cartesian(50 * 200, 1000, 256, 1)
grid = grid.centers(item="cell")[:, 0].reshape(1, 256, 1)
grid = torch.tensor(grid, dtype=torch.float)

S0_train = torch.from_numpy(S0_train)
S0_test = torch.from_numpy(S0_test)

trainY = torch.from_numpy(trainY)
testY = torch.from_numpy(testY)

# S0_train = torch.cat([S0_train.reshape(S0_train.shape[0], 256, 1), grid.repeat(S0_train.shape[0], 1, 1)], dim=2)
# S0_test = torch.cat([S0_test.reshape(S0_test.shape[0], 256, 1), grid.repeat(S0_test.shape[0], 1, 1)], dim=2)

S0_train = S0_train.reshape(S0_train.shape[0], 256, 1)
S0_test = S0_test.reshape(S0_test.shape[0], 256, 1)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(S0_train, trainY),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(S0_test, testY), batch_size=batch_size, shuffle=False
)

# # model
model = Net1d(modes, width)
print(model.count_params())
#
# ################################################################
# # training and evaluation
# ################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        # x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x.float())

        mse = F.mse_loss(out, y, reduction="mean")
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()  # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_mse = 0.0
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            # x, y = x, y.cuda()

            out = model(x.float())
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            mse = F.mse_loss(out, y, reduction="mean")

            test_mse += mse.item()

    test_mse /= len(test_loader)
    train_mse /= len(train_loader)
    train_l2 /= S0_train.shape[0]
    test_l2 /= S0_test.shape[0]

    t2 = default_timer()
    print(ep, t2 - t1, train_mse, train_l2, test_mse, test_l2)

torch.save(model, "level_0/global_approach/models/FNO")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(S0_train, trainY), batch_size=1, shuffle=False
)
index = 0

list_of_dict = []
with torch.no_grad():
    for x, y in train_loader:
        # x, y = x.cuda(), y.cuda()
        out = model(x.float())
        train_dict = {
            "S0": x.flatten().tolist(),
            "S": y.flatten().tolist(),
            "FNO_pred": out.tolist(),
        }
        list_of_dict.append(train_dict)

pred_train_df = pd.DataFrame(list_of_dict)
print("saving pred_train to csv")
pred_train_df.to_csv("level_0/global_approach/data/level_0_FNO_pred_train.csv")

index = 0
list_of_dict = []
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(S0_test, testY), batch_size=1, shuffle=False
)
with torch.no_grad():
    for x, y in test_loader:
        # x, y = x.cuda(), y.cuda()
        out = model(x.float())
        test_dict = {
            "S0": x.flatten().tolist(),
            "S": y.flatten().tolist(),
            "FNO_pred": out.tolist(),
        }
        list_of_dict.append(test_dict)

pred_test_df = pd.DataFrame(list_of_dict)
print("saving pred_test to csv")
pred_test_df.to_csv("level_0/global_approach/data/level_0_FNO_pred_test.csv")
