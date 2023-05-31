from torch import nn


class DeconvNet(nn.Module):
    def __init__(self, L, d):
        super(DeconvNet, self).__init__()
        self.L = L
        self.d = d
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.d), nn.Tanh(),
        )
        if self.L > 0:
            dense_L = L * [
                nn.Linear(in_features=self.d, out_features=self.d),
                nn.Tanh(),
            ]
            self.dense_L = nn.Sequential(*dense_L)
        self.ConvT = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.d,
                out_channels=self.d,
                kernel_size=(4, 4),
                stride=(5, 4),
                padding=(0, 0),
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                in_channels=self.d,
                out_channels=self.d,
                kernel_size=(4, 4),
                stride=(5, 4),
                padding=(0, 0),
            ),
            nn.Tanh(),
            nn.ConvTranspose2d(
                in_channels=self.d,
                out_channels=1,
                kernel_size=(5, 4),
                stride=(5, 4),
                padding=(0, 2),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ## Dense layers from (2) to (d)
        x = self.flatten(x)
        x = self.dense_1(x)
        if self.L > 0:
            self.dense_L(x)
        ## ConvTranspose2D from (1, 1, d) to (95, 60, 1)
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.ConvT(x)
        return x
