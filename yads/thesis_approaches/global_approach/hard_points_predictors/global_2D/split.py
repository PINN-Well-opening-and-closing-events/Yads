import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/2000_10_100_newtons_2D_global.csv")


train, test = train_test_split(df, test_size=0.50, random_state=42, shuffle=True)
print(train.shape, test.shape)
print(df["case"].value_counts())
train.to_csv("data/1000_10_100_newtons_2D_global_train.csv", index=False)
print(train["case"].value_counts())
test.to_csv("data/1000_10_100_newtons_2D_global_test.csv", index=False)
