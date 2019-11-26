import pandas as pd
import numpy as np
from pandas import ExcelWriter
import os
#%%
new_dir = "./Y_hat/"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

col = ['NN_16_2_linear', 'NN_16_2_relu', 'NN_16_4_2_linear', 'NN_16_4_2_relu', 'NN_16_8_4_2_linear', 'NN_16_8_4_2_relu']

for asset in range(1, 56):
    result_df = []
    for year in range(2006, 2016):
        file_name = './test_year{}/result_new/test_year{}_asset{}_yhat.csv'.format(year,year,asset)
        df = pd.read_csv(file_name, index_col=0)[col]
        result_df.append(df)
    result = pd.concat(result_df, 0).reset_index(drop=True)
    result.index = pd.date_range(start='2006-01-01', end='2015-12-01', freq='MS')
    result.to_csv(new_dir+'asset{}_yhat.csv'.format(asset))

