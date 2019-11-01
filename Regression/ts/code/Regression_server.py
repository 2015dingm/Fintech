import pandas as pd
import numpy as np
import xlwt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TKAgg")
print(matplotlib.get_backend())
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

df = pd.read_excel ('../../data/Return.xlsx')
df['TotRet'] = df['ExRet'] + df['Rf']
# print(df)

df_asset = []
for i in range(55):
    temp = df[df.ID == i+1]
    temp.reset_index(drop=True, inplace=True)
    df_asset.append(temp)
# print(df_asset)

def figure(y_true, y_predict, k, type):
    x = range(len(y_true))
    plt.figure(figsize=(10, 5))
    plt.ylabel("ExRet")
    plt.xlabel("time")
    plt.plot(x, y_true, label="true")
    plt.plot(x, y_predict, label="predict")
    plt.legend()
    plt.savefig('../../other_methods_cs/asset'+str(k+1)+'/regression_'+type+'.pdf', format='pdf')

def LinearReg(id):
    Data = pd.DataFrame(columns=['Ret-60', 'Ret-59', 'Ret-58', 'Ret-57', 'Ret-56',
                                 'Ret-55', 'Ret-54', 'Ret-53', 'Ret-52', 'Ret-51',
                                 'Ret-50', 'Ret-49', 'Ret-48', 'Ret-47', 'Ret-46',
                                 'Ret-45', 'Ret-44', 'Ret-43', 'Ret-42', 'Ret-41',
                                 'Ret-40', 'Ret-39', 'Ret-38', 'Ret-37', 'Ret-36',
                                 'Ret-35', 'Ret-34', 'Ret-33', 'Ret-32', 'Ret-31',
                                 'Ret-30', 'Ret-29', 'Ret-28', 'Ret-27', 'Ret-26',
                                 'Ret-25', 'Ret-24', 'Ret-23', 'Ret-22', 'Ret-21',
                                 'Ret-20', 'Ret-19', 'Ret-18', 'Ret-17', 'Ret-16',
                                 'Ret-15', 'Ret-14', 'Ret-13', 'Ret-12', 'Ret-11',
                                 'Ret-10', 'Ret-9', 'Ret-8', 'Ret-7', 'Ret-6',
                                 'Ret-5', 'Ret-4', 'Ret-3', 'Ret-2', 'Ret-1', 'ExRet'])

    nrow = df_asset[id].shape[0]
    for row in range(nrow - 60):
        temp = df_asset[id]
        x = temp.loc[row:row + 59, 'TotRet'].tolist()
        y = temp.loc[row + 60, 'ExRet']
        x.append(y)
        Data.loc[row] = x

    Data.to_csv('../../other_methods_cs/asset'+str(id+1)+'/regression.csv')
    nrow = Data.shape[0]
    valid = nrow - 72
    test = nrow - 12
    # print(valid)
    X_train = Data.iloc[:valid, :60]
    Y_train = Data.iloc[:valid, 60]

    X_valid = Data.iloc[valid: test, :60]
    Y_valid = Data.iloc[valid: test, 60]

    X_test = Data.iloc[test:, :60]
    Y_test = Data.iloc[test:, 60]
    # X_test = Data.iloc[valid:, :60]
    # Y_test = Data.iloc[valid:, 60]

    linear_reg = LinearRegression()
    linear_reg.fit(X_train, Y_train)

    Y_valid_predict = linear_reg.predict(X_valid)
    Y_predict = linear_reg.predict(X_test)
    train_score = linear_reg.score(X_train, Y_train)
    valid_score = linear_reg.score(X_valid, Y_valid)
    test_score = linear_reg.score(X_test, Y_test)
    mse_valid = mean_squared_error(Y_valid, Y_valid_predict)
    mse_test = mean_squared_error(Y_test, Y_predict)

    figure(Y_valid, Y_valid_predict, id, 'valid')
    figure(Y_test, Y_predict, id, 'test')

    print('Asset '+str(id)+'---------------------')
    print('Train score: ', train_score)
    print('Valid score: ', valid_score)
    print('Test score: ', test_score)


for id in range(55):
    LinearReg(id)








