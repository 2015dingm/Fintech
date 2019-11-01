import pandas as pd
import numpy as np
from pandas import ExcelWriter
import os
#%%
new_dir = "./Y_hat/"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

col=['AutoEncoder (1 components)','AutoEncoder (3 components)','PCA (1 components)','PCA (3 components)','PLS (1 components)','PLS (3 components)',
         'OLS regression','Lasso regression','Ridge regression','Elastic Net','Boosted regression Tree','Random Forrest']

for asset in range(1, 56):
    result_df = []
    for year in range(2006, 2016):
        file_name = './asset{}/result_new/asset{}_{}_yhat.csv'.format(asset,asset,year)
        df = pd.read_csv(file_name, index_col=0)[col]
        result_df.append(df)
    result = pd.concat(result_df, 0).reset_index(drop=True)
    result.index = pd.date_range(start='2006-01-01', end='2015-12-01', freq='MS')
    result.to_csv(new_dir+'asset{}_yhat.csv'.format(asset))

