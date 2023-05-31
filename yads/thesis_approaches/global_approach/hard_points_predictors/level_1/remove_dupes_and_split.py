import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/5000_10_100_newtons_110_100_P.csv",
    converters={"S": literal_eval, "P": literal_eval},
)


train, test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
print(train.shape, test.shape)
print(df["case"].value_counts())
train.to_csv("data/5000_10_100_newtons_110_100_P_train.csv", index=False)
print(train["case"].value_counts())
test.to_csv("data/5000_10_100_newtons_110_100_P_test.csv", index=False)
