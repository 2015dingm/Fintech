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
portfolio_5x5 = [['SMALL LoBETA', 'ME1 BETA2', 'ME1 BETA3', 'ME1 BETA4', 'SMALL HiBETA',
                 'ME2 BETA1', 'ME2 BETA2', 'ME2 BETA3', 'ME2 BETA4', 'ME2 BETA5',
                 'ME3 BETA1', 'ME3 BETA2', 'ME3 BETA3', 'ME3 BETA4', 'ME3 BETA5',
                 'ME4 BETA1', 'ME4 BETA2', 'ME4 BETA3', 'ME4 BETA4', 'ME4 BETA5',
                 'BIG LoBETA', 'ME5 BETA2', 'ME5 BETA3', 'ME5 BETA4', 'BIG HiBETA'],
                 ['SMALL LoVAR', 'ME1 VAR2', 'ME1 VAR3', 'ME1 VAR4', 'SMALL HiVAR',
                  'ME2 VAR1', 'ME2 VAR2', 'ME2 VAR3', 'ME2 VAR4', 'ME2 VAR5',
                  'ME3 VAR1', 'ME3 VAR2', 'ME3 VAR3', 'ME3 VAR4', 'ME3 VAR5',
                  'ME4 VAR1', 'ME4 VAR2', 'ME4 VAR3', 'ME4 VAR4', 'ME4 VAR5',
                  'BIG LoVAR', 'ME5 VAR2', 'ME5 VAR3', 'ME5 VAR4', 'BIG HiVAR'],
                 ['SMALL LoVAR', 'ME1 VAR2', 'ME1 VAR3', 'ME1 VAR4', 'SMALL HiVAR',
                  'ME2 VAR1', 'ME2 VAR2', 'ME2 VAR3', 'ME2 VAR4', 'ME2 VAR5',
                  'ME3 VAR1', 'ME3 VAR2', 'ME3 VAR3', 'ME3 VAR4', 'ME3 VAR5',
                  'ME4 VAR1', 'ME4 VAR2', 'ME4 VAR3', 'ME4 VAR4', 'ME4 VAR5',
                  'BIG LoVAR', 'ME5 VAR2', 'ME5 VAR3', 'ME5 VAR4', 'BIG HiVAR'],
                 ['SMALL LoAC', 'ME1 AC2', 'ME1 AC3', 'ME1 AC4', 'SMALL HiAC',
                  'ME2 AC1', 'ME2 AC2', 'ME2 AC3', 'ME2 AC4', 'ME2 AC5',
                  'ME3 AC1', 'ME3 AC2', 'ME3 AC3', 'ME3 AC4', 'ME3 AC5',
                  'ME4 AC1', 'ME4 AC2', 'ME4 AC3', 'ME4 AC4', 'ME4 AC5',
                  'BIG LoAC', 'ME5 AC2', 'ME5 AC3', 'ME5 AC4', 'BIG HiAC']]
# Ri = df_return_size_beta.loc[0:617]


# combinations
combi_rep = [[Mkt_rep], [Mkt_rep, SMB_rep, HML_rep], [Mkt_rep, SMB_rep, HML_rep, RMW_rep],
             [Mkt_rep, SMB_rep, HML_rep, CMA_rep], [Mkt_rep, SMB_rep, RMW_rep, CMA_rep],
             [Mkt_rep, SMB_rep, HML_rep, RMW_rep, CMA_rep]]
combi_upd = [[Mkt_upd], [Mkt_upd, SMB_upd, HML_upd], [Mkt_upd, SMB_upd, HML_upd, RMW_upd],
             [Mkt_upd, SMB_upd, HML_upd, CMA_upd], [Mkt_upd, SMB_upd, RMW_upd, CMA_upd],
             [Mkt_upd, SMB_upd, HML_upd, RMW_upd, CMA_upd]]
name = [['Mkt'], ['Mkt', 'SMB', 'HML'], ['Mkt', 'SMB', 'HML', 'RMW'], ['Mkt', 'SMB', 'HML', 'CMA'], ['Mkt', 'SMB', 'RMW', 'CMA'], ['Mkt', 'SMB', 'HML', 'RMW', 'CMA']]
num = [1, 3, 4, 4, 4, 5]
#zero = pd.Series(np.zeros(shape=618))


def regression(j, index, flag):
    a = []
    rbar = []
    R2 = []
    e = {}
    if flag == 1:   # replicate
        x = pd.concat(combi_rep[index], axis=1)
        X = sm.add_constant(x)
        K = num[index]
    if flag == 0:    # update
        x = pd.concat(combi_upd[index], axis=1)
        X = sm.add_constant(x)
        K = num[index]
    for i in portfolio_5x5[j]:
        if flag == 1:
            y = df_return_rep[i]-df_return_rep['RF']
            model = sm.OLS(y, X)
            results = model.fit()
            a.append(results.params['const'])
            rbar.append(y.mean() - Mkt_rep.mean())
            R2.append(results.rsquared)
            e[i] = results.resid
        if flag == 0:
            y = df_return_update[i]-df_return_update['RF']
            model = sm.OLS(y, X)
            results = model.fit()
            a.append(results.params['const'])
            rbar.append(y.mean()-Mkt_upd.mean())
            R2.append(results.rsquared)
            e[i] = results.resid
        # print(results.summary())
        # print(results.params['const'])
        # print(results.params)

    fbar = x.mean()
    fbar = np.mat(fbar).reshape(K, 1)
    # print(fbar)
    omega = np.mat(x.cov().values)
    # print(np.shape(omega))
    e = pd.DataFrame(e, index=np.arange(618))
    sigma = np.mat(e.cov().values).reshape(25, 25)
    # print(np.shape(sigma))

    a_abs = np.abs(a)
    a_square = np.square(a)
    a_var = np.var(a)
    a = np.mat(a).reshape(25, 1)

    rbar_abs = np.abs(rbar)
    rbar_square = np.square(rbar)
    rbar = np.mat(rbar).reshape(25, 1)

    R2 = np.array(R2)

    print(name[index])
    print('a_abs = ', a_abs.mean(), ', a_abs/r_abs = ', a_abs.mean()/rbar_abs.mean(), '\n'
          'a_squared/r_squared = ', a_square.mean()/rbar_square.mean(),
          ', a_var/a_squared = ', a_var.mean()/a_square.mean(), '\n'
          'R2 = ', R2.mean())
    # print(np.shape(a), np.shape(fbar))
    return a, fbar, omega, sigma, K



def GRS(a, fbar, omega, sigma, K, alpha=0.05):
    T = 618
    N = 25

    omega_mat = nlg.inv(omega)
    sigma_mat = nlg.inv(sigma)
    # print(np.shape(omega_mat), np.shape(sigma_mat))
    GRS = (T - N - K) * 1.0 / N * (a.T * sigma_mat * a) / (1 + fbar.T * omega_mat * fbar)[0][0]
    # F = stats.f.isf(alpha, N, T - N - K)  # 自由度为N,T-N-K，显著水平alpha%下的F分位值
    # p_value = 1 - 2 * abs(0.5 - stats.f.cdf(F, N, T - N - K))

    fdistribution = stats.f(N, T - N - K)  # build an F-distribution object
    p_value = 1 - fdistribution.cdf(GRS)
    # p_value = 1 - stats.f.cdf(GRS, N, T - N - K)

    print('GRS =', GRS, ', pvalue =', p_value)
    print('\n')





for j in range(4):
    df_list = [df_return_size_beta, df_return_size_var, df_return_size_rvar, df_return_size_ac]
    name_list = ['25 size-beta portfolios', '25 size-var portfolios', '25 size-rvar portfolios', '25 size-ac portfolios']
    df_return_rep = df_list[j].loc[0:617]
    df_return_update = df_list[j].loc[0:665]
    print('==================================')
    print(name_list[j], ':')
    for n in range(0, 6):
        print('-----------')
        print('Replicate:')
        a, fbar, omega, sigma, K = regression(j, n, 1)
        GRS(a, fbar, omega, sigma, K)
    for n in range(0, 6):
        print('-----------')
        print('Update:')
        a, fbar, omega, sigma, K = regression(j, n, 0)
        GRS(a, fbar, omega, sigma, K)

