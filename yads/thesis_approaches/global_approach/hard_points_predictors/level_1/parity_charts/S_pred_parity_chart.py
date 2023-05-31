import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval

test = pd.read_csv(
    "../data/5000_10_100_newtons_sirious_train.csv", converters={"S": literal_eval}
)

S_predict = pickle.load(open("../models/S_predict_model_hard_xgboost.pkl", "rb"))
S_predict_scaler = pickle.load(open("../models/S_predict_scaler_xgboost.pkl", "rb"))

test["log_q"] = np.log10(-test["q"])

test_gentle = test[test["case"] == "gentle"]
S_pred_gentle = S_predict.predict(
    S_predict_scaler.transform(test_gentle[["log_q", "dt_init"]])
)
S_pred_gentle = [np.linalg.norm(S_pred_gentle[i]) for i in range(len(S_pred_gentle))]
S_gentle = [
    np.linalg.norm(test_gentle["S"].values[i]) for i in range(len(S_pred_gentle))
]

test_hard = test[test["case"] == "hard"]
S_pred_hard = S_predict.predict(
    S_predict_scaler.transform(test_hard[["log_q", "dt_init"]])
)
S_pred_hard = [np.linalg.norm(S_pred_hard[i]) for i in range(len(S_pred_hard))]
S_hard = [np.linalg.norm(test_hard["S"].values[i]) for i in range(len(S_pred_hard))]

fig = plt.figure(figsize=(10, 10))
plt.scatter(S_gentle, S_pred_gentle, s=2, label="gentle cases")
plt.scatter(S_hard, S_pred_hard, s=2, label="hard cases")
plt.xlabel("True Saturation L2 norm")
plt.ylabel("Predicted Saturation L2 norm")
plt.xlim(0)
plt.ylim(0)
plt.legend()
plt.show()
