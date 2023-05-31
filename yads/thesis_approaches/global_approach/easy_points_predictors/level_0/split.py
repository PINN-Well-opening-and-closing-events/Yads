import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "global_approach/data/level_0_5000.csv",
    converters={"S": literal_eval, "P": literal_eval, "S0": literal_eval},
)


train, test = train_test_split(df, test_size=0.80, random_state=42, shuffle=True)
print(train.shape, test.shape)
train.to_csv("data/train_1000.csv", index=False)
test.to_csv("data/test_4000.csv", index=False)
