import numpy as np
import pandas as pd
from ast import literal_eval
import pickle

# df = pd.read_csv("data/2000_10_100_newtons_2D_global.csv.csv",
#                  converters={'S': literal_eval, 'P': literal_eval})
# df['log_q'] = np.log10(-df['q'])

train = pd.read_csv(
    "data/1000_10_100_newtons_2D_global_train.csv",
    converters={"S": literal_eval, "P": literal_eval},
)
test = pd.read_csv(
    "data/1000_10_100_newtons_2D_global_test.csv",
    converters={"S": literal_eval, "P": literal_eval},
)
train["log_q"] = np.log10(-train["q"])
test["log_q"] = np.log10(-test["q"])

classifier = pickle.load(open("models/case_classifier_model_xgboost.pkl", "rb"))
classifier_scaler = pickle.load(open("models/case_classifier_scaler_xgboost.pkl", "rb"))


def find_optimal_dt(df, model, scaler):
    qts = df[["log_q", "dt_init"]]
    optimal_dts = []
    for i in range(len(qts)):
        if df["case"][i] == "giga_hard":
            is_optimized = False
            qt = qts.iloc[i].values
            qt[1] *= 0.95
            qt_scaled = scaler.transform(qt.reshape(1, -1))
            while not is_optimized:
                qt_pred = model.predict(qt_scaled)[0]
                if qt_pred == 0:
                    optimal_dts.append(qt[1])
                    is_optimized = True
                elif qt_pred == 1:
                    qt[1] *= 0.95
                    qt_scaled = scaler.transform(qt.reshape(1, -1))
                else:
                    print(f"unknown class wtf for case {i}")
        elif df["case"][i] == "gentle" or df["case"][i] == "hard":
            optimal_dts.append(0)
    return optimal_dts


train["optimal_dt"] = find_optimal_dt(train, classifier, classifier_scaler)
train.to_csv("data/1000_10_100_newtons_2D_global_train.csv")
print(train[train["case"] == "giga_hard"].describe())
test["optimal_dt"] = find_optimal_dt(test, classifier, classifier_scaler)
test.to_csv("data/1000_10_100_newtons_2D_global_test.csv")
print(test[test["case"] == "giga_hard"].describe())
