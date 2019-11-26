import pandas as pd
import numpy as np
from pandas import ExcelWriter
import os
from R2OOS import *
from CWtest import *

def historical_average(df):
    '''
        not including this month
    '''
    df_new = pd.DataFrame(None, index=df.index, columns=df.columns)
    for i in range(len(df)):
        df_new.iloc[i] = df[:i].mean()
    return df_new

#%%
new_dir = "./result_sur/"
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

ret=pd.read_excel('./data/Return.xlsx','Return')

col = ['NN_16_2_linear', 'NN_16_2_relu', 'NN_16_4_2_linear', 'NN_16_4_2_relu', 'NN_16_8_4_2_linear', 'NN_16_8_4_2_relu']
#define the variables that we want to achieve
Table_r2 = pd.DataFrame(columns=col, index=np.arange(1,56))
Table_cw = Table_r2.copy()

# join Y_hat from DL and OM
for asset in range(1, 56):
    dl_file = './sur/Y_hat/asset{}_yhat.csv'.format(asset)
    # om_file = './other_methods_cs/Y_hat/asset{}_yhat.csv'.format(asset)
    df = pd.read_csv(dl_file, index_col=0)
    # om_df = pd.read_csv(om_file, index_col=0)
    # df = pd.concat([om_df, dl_df], 1)
    df.to_csv(new_dir+'asset{}_yhat.csv'.format(asset))
# y_bar
    ret_sub = ret[ret['ID']==asset]
    ret_sub.set_index('YM', inplace=True)
    ha = historical_average(ret_sub)
    r_bar = ha[ha.index>=200601]['ExRet'].values # make sure all returns are after 2000-01
# y_real
    r_real = ret_sub[ret_sub.index>=200601]['ExRet'].values

    for col in df.columns:
        r_hat = df[col].values
        Table_r2.loc[[asset], [col]] = R2OOS(r_real,r_hat,r_bar)*100
        Table_cw.loc[[asset], [col]] = CWtest(r_real,r_hat,r_bar,4)[1]  #p_val

writer = ExcelWriter(new_dir+'result1016_sur.xlsx')
Table_r2.to_excel(writer, 'R2')
Table_cw.to_excel(writer, 'CW')
writer.save()
