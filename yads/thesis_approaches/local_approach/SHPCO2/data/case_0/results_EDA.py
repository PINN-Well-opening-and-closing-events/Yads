import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv("results/quantification_4_test_0_1_0.csv",
                   converters={'S_i_plus_1_DD': literal_eval, 'P_i_plus_1_DD': literal_eval},
                                       sep="\t")

print(test.columns)
print(np.array(test['S_i_plus_1_DD'].loc[0]).shape)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(np.array(test['S_i_plus_1_DD'].loc[0]).reshape(95, 60).T)
ax1.invert_yaxis()
ax2.imshow(np.array(test['P_i_plus_1_DD'].loc[0]).reshape(95, 60).T)
ax2.invert_yaxis()
plt.show()