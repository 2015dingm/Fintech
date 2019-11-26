import numpy as np
import pandas as pd
import numpy.linalg as nlg
import scipy.stats as stats

R_SIZE_VAR = 'data/25_Size_Var_Portfolios.csv'
R_SIZE_VAR_BM = 'data/25_Size_Var_Portfolios(BM).csv'
R_SIZE_VAR_OP = 'data/25_Size_Var_Portfolios(OP).csv'
R_SIZE_VAR_INV = 'data/25_Size_Var_Portfolios(Inv).csv'
R_SIZE_VAR_VAR = 'data/25_Size_Var_Portfolios(Var).csv'
df_return_size_var = pd.read_csv(R_SIZE_VAR)
df_return_size_var_bm = pd.read_csv(R_SIZE_VAR_BM)
df_return_size_var_op = pd.read_csv(R_SIZE_VAR_OP)
df_return_size_var_inv = pd.read_csv(R_SIZE_VAR_INV)
df_return_size_var_var = pd.read_csv(R_SIZE_VAR_VAR)

portfolio_5x5 = ['SMALL LoVAR', 'ME1 VAR2', 'ME1 VAR3', 'ME1 VAR4', 'SMALL HiVAR',
                  'ME2 VAR1', 'ME2 VAR2', 'ME2 VAR3', 'ME2 VAR4', 'ME2 VAR5',
                  'ME3 VAR1', 'ME3 VAR2', 'ME3 VAR3', 'ME3 VAR4', 'ME3 VAR5',
                  'ME4 VAR1', 'ME4 VAR2', 'ME4 VAR3', 'ME4 VAR4', 'ME4 VAR5',
                  'BIG LoVAR', 'ME5 VAR2', 'ME5 VAR3', 'ME5 VAR4', 'BIG HiVAR']

# print(df_return_size_beta[0:666])

# replicate
print('================================')
print('Replicate')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_var.loc[0:618, portfolio_5x5[i]]-df_return_size_var.loc[0:618, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_var.loc[0:618, portfolio_5x5[i]]-df_return_size_var.loc[0:618, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_bm.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_op.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_inv.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('Var:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_var.loc[0:618, portfolio_5x5[i]].mean())




# update

print('================================')
print('Update')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_var.loc[0:666, portfolio_5x5[i]]-df_return_size_var.loc[0:666, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_var.loc[0:666, portfolio_5x5[i]]-df_return_size_var.loc[0:666, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_bm.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_op.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_inv.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('Var:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_var_var.loc[0:666, portfolio_5x5[i]].mean())
