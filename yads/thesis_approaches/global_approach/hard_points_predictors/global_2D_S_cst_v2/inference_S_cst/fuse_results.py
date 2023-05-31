import pandas as pd

filenames = [
    "result_quantification_test_0_3.csv",
    "result_quantification_test_20_23.csv",
    "result_quantification_test_40_43.csv",
    "result_quantification_test_100_103.csv",
    "result_quantification_test_200_203.csv",
    "result_quantification_test_300_305.csv",
    "result_quantification_test_400_405.csv",
    "result_quantification_test_500_503.csv",
    "result_quantification_test_600_603.csv",
]

filenames_2 = [
    "result_quantification_newton_sup_200_0_3.csv",
    "result_quantification_newton_sup_200_4_7.csv",
    "result_quantification_newton_sup_200_8_11.csv",
    "result_quantification_newton_sup_200_12_17.csv",
    "result_quantification_newton_sup_200_18_21.csv",
]
df_list = []

for f in filenames_2:
    df = pd.read_csv("results/" + f, sep="\t")
    print(df.columns)
    df_list.append(df)

big_df = pd.concat(df_list)
print(big_df.columns)
big_df.to_csv(
    "result_quantification_test_S_cst_nb_newton_sup_200.csv", sep="\t", index=False
)
