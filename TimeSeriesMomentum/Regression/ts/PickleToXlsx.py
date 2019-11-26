import pandas as pd
import numpy as np
from pandas import ExcelWriter
import os
#%%
col = ['AutoEncoder (1 components)','AutoEncoder (3 components)','PCA (1 components)','PCA (3 components)','PLS (1 components)','PLS (3 components)',
       'OLS regression','Lasso regression','Ridge regression','Elastic Net','Boosted regression Tree','Random Forrest']

new_dir = "./result_raw/"

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for asset in range(1,56):
    result_df = []
    for year in range(2000, 2014):
        file_name = './result/_'+data_name+str(year)+'_rmse.pickle'
        d = pd.read_pickle(file_name)
        df = pd.DataFrame(d).T
        # df.columns = ['rx0', 'rx1', 'rx2', 'rx3']
        result_df.append(df.values)
    result = np.stack(result_df, 0)
    rmse_array = result.mean(0)
    rmse_df = pd.DataFrame(rmse_array,index=df.index,columns=df.columns)
    rmse_df.to_csv(new_dir+"{}Rmse_cs.csv".format(data_name))

