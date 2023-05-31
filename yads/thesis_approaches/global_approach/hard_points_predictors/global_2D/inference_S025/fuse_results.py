import pandas as pd

filenames = [
    "result_quantification_test_20_23.csv",
    "result_quantification_test_40_43.csv",
    "result_quantification_test_100_103.csv",
    "result_quantification_test_200_203.csv",
]

df_list = []

for f in filenames:
    df = pd.read_csv("results/" + f, sep="\t")
    print(df.columns)
    df_list.append(df)

big_df = pd.concat(df_list)
print(big_df.columns)
big_df.to_csv("result_quantification_test_16_samples.csv", sep="\t", index=False)
