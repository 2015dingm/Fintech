import numpy as np
import pandas as pd
import numpy.linalg as nlg
import scipy.stats as stats
import statsmodels.api as sm

R_SIZE_BETA = 'data/25_Size_Beta_Portfolios.csv'
R_SIZE_VAR = 'data/25_Size_Var_Portfolios.csv'
R_SIZE_RVAR = 'data/25_Size_RVar_Portfolios.csv'
R_SIZE_AC = 'data/25_Size_AC_Portfolios.csv'
FACTOR = 'data/5_Factor_2x3.csv'
df_return_size_beta = pd.read_csv(R_SIZE_BETA)
df_return_size_var = pd.read_csv(R_SIZE_VAR)
df_return_size_rvar = pd.read_csv(R_SIZE_RVAR)
df_return_size_ac = pd.read_csv(R_SIZE_AC)
df_factor = pd.read_csv(FACTOR)



# replica

# factor
Mkt_rep = df_factor.loc[0:617, 'Mkt-RF']
SMB_rep = df_factor.loc[0:617, 'SMB']
HML_rep = df_factor.loc[0:617, 'HML']
RMW_rep = df_factor.loc[0:617, 'RMW']
CMA_rep = df_factor.loc[0:617, 'CMA']

Mkt_upd = df_factor.loc[0:665, 'Mkt-RF']
SMB_upd = df_factor.loc[0:665, 'SMB']
HML_upd = df_factor.loc[0:665, 'HML']
RMW_upd = df_factor.loc[0:665, 'RMW']
CMA_upd = df_factor.loc[0:665, 'CMA']
# print(Mkt)

# return
portfolio_5x5 = ['SMALL LoAC', 'ME1 AC2', 'ME1 AC3', 'ME1 AC4', 'SMALL HiAC',
                  'ME2 AC1', 'ME2 AC2', 'ME2 AC3', 'ME2 AC4', 'ME2 AC5',
                  'ME3 AC1', 'ME3 AC2', 'ME3 AC3', 'ME3 AC4', 'ME3 AC5',
                  'ME4 AC1', 'ME4 AC2', 'ME4 AC3', 'ME4 AC4', 'ME4 AC5',
                  'BIG LoAC', 'ME5 AC2', 'ME5 AC3', 'ME5 AC4', 'BIG HiAC']

df_return_rep = df_return_size_ac.loc[0:617]
df_return_update = df_return_size_ac.loc[0:665]
# print(df_return_update)
# Ri = df_return_size_beta.loc[0:617]


# combinations
combi_rep = [[Mkt_rep, SMB_rep, HML_rep], [Mkt_rep, SMB_rep, HML_rep, CMA_rep],
             [Mkt_rep, SMB_rep, RMW_rep, CMA_rep],
             [Mkt_rep, SMB_rep, HML_rep, RMW_rep, CMA_rep]]
combi_upd = [[Mkt_upd, SMB_upd, HML_upd], [Mkt_upd, SMB_upd, HML_upd, CMA_upd],
             [Mkt_upd, SMB_upd, RMW_upd, CMA_upd],
             [Mkt_upd, SMB_upd, HML_upd, RMW_upd, CMA_upd]]
name = [['Mkt', 'SMB', 'HML'],
        ['Mkt', 'SMB', 'HML', 'CMA'],
        ['Mkt', 'SMB', 'RMW', 'CMA'],     # for 'Mkt', 'SMB', 'RMW', 'CMA'
        ['Mkt', 'SMB', 'HML', 'RMW', 'CMA']]      # 'HMLO'
num = [3, 4, 4, 5]
#zero = pd.Series(np.zeros(shape=618))


def regression(portfolio, index, flag):

    if flag == 1:  # replicate
        x = pd.concat(combi_rep[index], axis=1)
        X = sm.add_constant(x)
        K = num[index]
    if flag == 0:   # update
        x = pd.concat(combi_upd[index], axis=1)
        X = sm.add_constant(x)
        K = num[index]

    if flag == 1:
        y = df_return_rep[portfolio]-df_return_rep['RF']
        model = sm.OLS(y, X)
        results = model.fit()
        print('Portfolio:', portfolio)
        print(results.summary())

    if flag == 0:
        y = df_return_update[portfolio]-df_return_update['RF']
        model = sm.OLS(y, X)
        results = model.fit()
        print('Portfolio:', portfolio)
        print(results.summary())





# replicate
for index in range(4):
    print('==================')
    print(name[index])
    for portfolio in portfolio_5x5:
        regression(portfolio, index, 1)

# update
for index in range(4):
    print('==================')
    print(name[index])
    for portfolio in portfolio_5x5:
        regression(portfolio, index, 0)


