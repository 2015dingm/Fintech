import numpy as np
import pandas as pd
import numpy.linalg as nlg
import scipy.stats as stats

R_SIZE_BETA = 'data/25_Size_Beta_Portfolios.csv'
R_SIZE_BETA_BM = 'data/25_Size_Beta_Portfolios(BM).csv'
R_SIZE_BETA_OP = 'data/25_Size_Beta_Portfolios(OP).csv'
R_SIZE_BETA_INV = 'data/25_Size_Beta_Portfolios(Inv).csv'
R_SIZE_BETA_PRIOR = 'data/25_Size_Beta_Portfolios(PriorBeta).csv'
df_return_size_beta = pd.read_csv(R_SIZE_BETA)
df_return_size_beta_bm = pd.read_csv(R_SIZE_BETA_BM)
df_return_size_beta_op = pd.read_csv(R_SIZE_BETA_OP)
df_return_size_beta_inv = pd.read_csv(R_SIZE_BETA_INV)
df_return_size_beta_prior = pd.read_csv(R_SIZE_BETA_PRIOR)

portfolio_5x5 = ['SMALL LoBETA', 'ME1 BETA2', 'ME1 BETA3', 'ME1 BETA4', 'SMALL HiBETA',
                 'ME2 BETA1', 'ME2 BETA2', 'ME2 BETA3', 'ME2 BETA4', 'ME2 BETA5',
                 'ME3 BETA1', 'ME3 BETA2', 'ME3 BETA3', 'ME3 BETA4', 'ME3 BETA5',
                 'ME4 BETA1', 'ME4 BETA2', 'ME4 BETA3', 'ME4 BETA4', 'ME4 BETA5',
                 'BIG LoBETA', 'ME5 BETA2', 'ME5 BETA3', 'ME5 BETA4', 'BIG HiBETA']

# print(df_return_size_beta[0:666])

# replicate
print('================================')
print('Replicate')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_beta.loc[0:618, portfolio_5x5[i]]-df_return_size_beta.loc[0:618, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_beta.loc[0:618, portfolio_5x5[i]]-df_return_size_beta.loc[0:618, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_bm.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_op.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_inv.loc[0:618, portfolio_5x5[i]].mean())
print('---------------------')
print('Prior Beta:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_prior.loc[0:618, portfolio_5x5[i]].mean())




# update

print('================================')
print('Update')
print('Mean:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_beta.loc[0:666, portfolio_5x5[i]]-df_return_size_beta.loc[0:666, 'RF']).mean())
print('---------------------')
print('SD:')
for i in range(25):
   print(portfolio_5x5[i], ':', (df_return_size_beta.loc[0:666, portfolio_5x5[i]]-df_return_size_beta.loc[0:666, 'RF']).std())
print('---------------------')
print('B/M:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_bm.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('OP:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_op.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('Inv:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_inv.loc[0:666, portfolio_5x5[i]].mean())
print('---------------------')
print('Prior Beta:')
for i in range(25):
   print(portfolio_5x5[i], ':', df_return_size_beta_prior.loc[0:666, portfolio_5x5[i]].mean())
