import numpy as np
import pandas as pd
import numpy.linalg as nlg
import scipy.stats as stats

R_SIZE_AC = 'data/25_Size_AC_Portfolios.csv'
R_SIZE_AC_BM = 'data/25_Size_AC_Portfolios(BM).csv'
R_SIZE_AC_OP = 'data/25_Size_AC_Portfolios(OP).csv'
R_SIZE_AC_INV = 'data/25_Size_AC_Portfolios(Inv).csv'
R_SIZE_AC_AC = 'data/25_Size_AC_Portfolios(AC).csv'
df_return_size_ac = pd.read_csv(R_SIZE_AC)
df_return_size_ac_bm = pd.read_csv(R_SIZE_AC_BM)
df_return_size_ac_op = pd.read_csv(R_SIZE_AC_OP)
df_return_size_ac_inv = pd.read_csv(R_SIZE_AC_INV)
df_return_size_ac_ac = pd.read_csv(R_SIZE_AC_AC)

portfolio_5x5 = ['SMALL LoAC', 'ME1 AC2', 'ME1 AC3', 'ME1 AC4', 'SMALL HiAC',
                  'ME2 AC1', 'ME2 AC2', 'ME2 AC3', 'ME2 AC4', 'ME2 AC5',
                  'ME3 AC1', 'ME3 AC2', 'ME3 AC3', 'ME3 AC4', 'ME3 AC5',
                  'ME4 AC1', 'ME4 AC2', 'ME4 AC3', 'ME4 AC4', 'ME4 AC5',
                  'BIG LoAC', 'ME5 AC2', 'ME5 AC3', 'ME5 AC4', 'BIG HiAC']

# print(df_return_size_beta[0:666])

# replicate
print('================================')
print('Replicate')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_ac.loc[0:618, portfolio_5x5[i]]-df_return_size_ac.loc[0:618, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_ac.loc[0:618, portfolio_5x5[i]]-df_return_size_ac.loc[0:618, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_bm.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_op.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_inv.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('AC:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_ac.loc[0:618, portfolio_5x5[i]].mean())




# update

print('================================')
print('Update')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_ac.loc[0:666, portfolio_5x5[i]]-df_return_size_ac.loc[0:666, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_ac.loc[0:666, portfolio_5x5[i]]-df_return_size_ac.loc[0:666, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_bm.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_op.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_inv.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('AC:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_ac_ac.loc[0:666, portfolio_5x5[i]].mean())
