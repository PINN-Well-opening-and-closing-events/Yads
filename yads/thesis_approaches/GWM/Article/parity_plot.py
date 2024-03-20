import pandas as pd
import torch
import pickle
from inference.models.FNO import FNO2d, UnitGaussianNormalizer
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

S_model = model = FNO2d(modes1=12, modes2=12, width=64, n_features=4)
S_model.load_state_dict(
    torch.load(
        "inference/models/GWM_3100_checkpoint_2500.pt",
        map_location=torch.device("cpu"),
    )["model"]
)

q_normalizer = pickle.load(open("inference/models/GWM_q_normalizer.pkl", "rb"))
P_imp_normalizer = pickle.load(open("inference/models/GWM_P_imp_normalizer.pkl", "rb"))
dt_normalizer = pickle.load(open("inference/models/GWM_dt_normalizer.pkl", "rb"))

res_length = 9
well_loc = 40

df = pd.read_csv(
    "data/test_gwm_dataset_3100.csv",
    sep="\t",
    converters={"S": literal_eval, "P_imp": literal_eval, "S0": literal_eval},
    nrows=1000,
)
nb_data = len(df)
df["log_q"] = -np.log10(-df["q"])
df["log_dt"] = np.log(df["dt"])

# create features maps
# q is map of 0 everywhere except at well loc
q_flat_zeros = [np.zeros((res_length * res_length)) for _ in range(len(df["log_q"]))]
for i in range(len(q_flat_zeros)):
    q_flat_zeros[i][well_loc] = df["log_q"][i]

S0_flat_zeros = [np.full((res_length * res_length), S0) for S0 in df["S0"].values]

q = torch.from_numpy(
    np.array([np.reshape(q_f, (res_length, res_length, 1)) for q_f in q_flat_zeros])
)
dt = torch.from_numpy(
    np.array([np.full((res_length, res_length, 1), qs) for qs in df["log_dt"]])
)
S0 = torch.from_numpy(
    np.array([np.array(v).reshape(res_length, res_length, 1) for v in df["S0"].values])
)
P_imp = torch.from_numpy(
    np.array(
        [
            np.array(np.log10(v)).reshape(res_length, res_length, 1)
            for v in df["P_imp"].values
        ]
    )
)


q = q_normalizer.encode(q)
dt = dt_normalizer.encode(dt)
P_imp = P_imp_normalizer.encode(P_imp)

# concat all features maps to (Nsample, 9, 9, 4)
x = torch.cat([q, dt, S0, P_imp], 3)


y = df["S"]
y = torch.from_numpy(
    np.array([np.array(v).reshape(res_length, res_length, 1) for v in y.values])
)

x = x.float()
y = y.float()

y_pred = np.reshape(model(x).detach().numpy(), (nb_data, res_length * res_length))
y_true = np.reshape(y.detach().numpy(), (nb_data, res_length * res_length))


y_true_norms_test = np.linalg.norm(y_true, ord=2, axis=1)
y_pred_norms_test = np.linalg.norm(y_pred, ord=2, axis=1)

df = pd.read_csv(
    "data/train_gwm_dataset_3100.csv",
    sep="\t",
    converters={"S": literal_eval, "P_imp": literal_eval, "S0": literal_eval},
    nrows=1000,
)
nb_data = len(df)
df["log_q"] = -np.log10(-df["q"])
df["log_dt"] = np.log(df["dt"])

# create features maps
# q is map of 0 everywhere except at well loc
q_flat_zeros = [np.zeros((res_length * res_length)) for _ in range(len(df["log_q"]))]
for i in range(len(q_flat_zeros)):
    q_flat_zeros[i][well_loc] = df["log_q"][i]

S0_flat_zeros = [np.full((res_length * res_length), S0) for S0 in df["S0"].values]

q = torch.from_numpy(
    np.array([np.reshape(q_f, (res_length, res_length, 1)) for q_f in q_flat_zeros])
)
dt = torch.from_numpy(
    np.array([np.full((res_length, res_length, 1), qs) for qs in df["log_dt"]])
)
S0 = torch.from_numpy(
    np.array([np.array(v).reshape(res_length, res_length, 1) for v in df["S0"].values])
)
P_imp = torch.from_numpy(
    np.array(
        [
            np.array(np.log10(v)).reshape(res_length, res_length, 1)
            for v in df["P_imp"].values
        ]
    )
)


q = q_normalizer.encode(q)
dt = dt_normalizer.encode(dt)
P_imp = P_imp_normalizer.encode(P_imp)

# concat all features maps to (Nsample, 9, 9, 4)
x = torch.cat([q, dt, S0, P_imp], 3)


y = df["S"]
y = torch.from_numpy(
    np.array([np.array(v).reshape(res_length, res_length, 1) for v in y.values])
)

x = x.float()
y = y.float()

y_pred = np.reshape(model(x).detach().numpy(), (nb_data, res_length * res_length))
y_true = np.reshape(y.detach().numpy(), (nb_data, res_length * res_length))


y_true_norms_train = np.linalg.norm(y_true, ord=2, axis=1)
y_pred_norms_train = np.linalg.norm(y_pred, ord=2, axis=1)

fig = plt.figure(figsize=(10, 10))
plt.scatter(
    x=y_true_norms_test, y=y_pred_norms_test, color=(68 / 255, 114 / 255, 196 / 255)
)
plt.scatter(x=y_true_norms_train, y=y_pred_norms_train, color="orange")

# plt.yscale('log')
# plt.xscale('log')
plt.xlabel(r"S solution L2 norm", size=20)
plt.ylabel(r"S predicted L2 norm", size=20)
# plt.axis('square')
plt.show()
