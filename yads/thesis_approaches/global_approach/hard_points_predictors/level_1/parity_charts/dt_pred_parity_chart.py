import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

test = pd.read_csv("../data/5000_10_100_newtons_sirious_test.csv")

dt_predict = pickle.load(open("../models/dt_predict_model_xgboost.pkl", "rb"))
dt_predict_scaler = pickle.load(open("../models/dt_predict_scaler_xgboost.pkl", "rb"))

test["log_q"] = np.log10(-test["q"])

test_giga_hard = test[test["case"] == "giga_hard"]

dt_pred_giga_hard = 10 ** dt_predict.predict(
    dt_predict_scaler.transform(test_giga_hard[["log_q", "dt_init"]].to_numpy())
)

fig = plt.figure(figsize=(10, 10))
plt.scatter(np.log(test_giga_hard["optimal_dt"]), np.log(dt_pred_giga_hard), s=2)
plt.xlabel("Optimal log10(dt)")
plt.ylabel("Predicted log10(dt)")
# plt.xlim(0)
# plt.ylim(0)
plt.show()
