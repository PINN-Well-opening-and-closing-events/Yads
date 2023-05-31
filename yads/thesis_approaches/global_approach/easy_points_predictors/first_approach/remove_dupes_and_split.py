import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/train_5000.csv", converters={"S": literal_eval, "P": literal_eval}
)

nb_dupes = df.duplicated(subset=["q", "total_time"]).sum()
print(nb_dupes)
# df = df.drop_duplicates(subset=['q', 'total_time'])


train, test = train_test_split(df, test_size=0.80, random_state=42, shuffle=True)
print(train.shape, test.shape)
# save the train and test file
# again using the '\t' separator to create tab-separated-values files
train.to_csv("../predictors/first_approach/data/train_1000.csv", index=False)
test.to_csv("../predictors/first_approach/data/test_4000.csv", index=False)
