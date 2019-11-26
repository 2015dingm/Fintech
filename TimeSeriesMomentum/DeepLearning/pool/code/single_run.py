'''
    This is for one asset- Deep learning model
'''
from script import col, teststart_list, feature_num_list, test_year
#%%
from NN_function_ts import *
from tuning import nb_epoch, batch_size, param_grid, param_alpha, l1_ratio
#%%
# fix random seed
from numpy.random import seed
seed(2019)
from tensorflow import set_random_seed
set_random_seed(2019)
#%%
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def get_data_one_asset(test_year, test_start, X, Y):
    '''
    test data = 12 m in test_year
    validation data = 60 m before test_year
    train data = all data before validation
    These three dataset has no intersection

    :param test_year: use data of 12 m in test_year as testing data
    :param X: features of bond dl with shape:(405, 63) 198203-201511
    :param Y: excess return of n_assets assets with shape:(405, 4) 198204-201512
    :return: X_train, X_val, X_train_val, X_test, Y_train, Y_val, Y_train_val, Y_test
    '''
    # test_sart[globalize] index of start of test data set, -1 for python start @ 0
    test_year_start = test_start + (test_year-2006)*12
    test_end = test_year_start + 12           # index of end of test data set, length of test = 12 m
    valid_start = test_year_start - 60        # index of start of validation data set

    ################################
    # X transformation
    X_train = X[:valid_start, :]
    X_val = X[valid_start:test_year_start, :]
    X_train_val = X[:test_year_start, :]
    X_test = X[test_year_start:test_end, :]
    # standardization for cross validation
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_train_val_std = scaler.transform(X_train_val)
    # standardization for final training as we train on train_val data to predict
    scaler2 = StandardScaler()
    scaler2.fit(X_train_val)
    X_test_std = scaler2.transform(X_test)

    ################################
    # Y transformation
    Y = Y / 100.0
    Y_train = Y[:valid_start]
    Y_val = Y[valid_start:test_year_start]
    Y_train_val = Y[:test_year_start]
    Y_test = Y[test_year_start:test_end]

    return X_train_std, X_val_std, X_train_val_std, X_test_std, Y_train, Y_val, Y_train_val, Y_test

# get training data from 55 assets
def get_training_data(test_year, X, Y):
    X_train_list = []
    X_train_val_list = []
    X_val_list = []
    Y_train_list = []
    Y_train_val_list = []
    Y_val_list = []

    for asset in range(1, 56):
        feature_num = 0
        for i in range(asset):
            feature_num = feature_num + feature_num_list[i]
        feature_start = feature_num - feature_num_list[asset-1]
        feature_end = feature_num

        X_sub = X.iloc[feature_start:feature_end, :].values
        Y_sub = Y[feature_start:feature_end].values

        test_start = teststart_list[asset - 1]

        X_train_, X_val_, X_train_val_, X_test_, Y_train_, Y_val_, Y_train_val_, Y_test_ = get_data_one_asset(test_year, test_start, X_sub, Y_sub)
        X_train_list.append(X_train_)
        X_train_val_list.append(X_train_val_)
        X_val_list.append(X_val_)
        Y_train_list.append(Y_train_)
        Y_train_val_list.append(Y_train_val_)
        Y_val_list.append(Y_val_)

        # if asset == asset_i:
        #     X_val = X_val_
        #     X_test = X_test_
        #     Y_val = Y_val_
        #     Y_test = Y_test_
        # print('X_train_val shape:{}\nY_train_val shape:{}\n'.format(X_train_val.shape, Y_train_val.shape))

    X_train = np.concatenate(X_train_list, axis=0)
    X_train_val = np.concatenate(X_train_val_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    Y_train_val = np.concatenate(Y_train_val_list, axis=0)
    Y_val = np.concatenate(Y_val_list, axis=0)

    return X_train, X_train_val, X_val, Y_train, Y_train_val, Y_val

def get_testing_data(test_year, X, Y, asset):
    feature_num = 0
    for i in range(asset):
        feature_num = feature_num + feature_num_list[i]
    feature_start = feature_num - feature_num_list[asset - 1]
    feature_end = feature_num

    X_sub = X.iloc[feature_start:feature_end, :].values
    Y_sub = Y[feature_start:feature_end].values

    test_start = teststart_list[asset - 1]

    X_train_, X_val_, X_train_val_, X_test_, Y_train_, Y_val_, Y_train_val_, Y_test_ = get_data_one_asset(test_year, test_start, X_sub, Y_sub)

    X_test = X_test_
    Y_test = Y_test_

    return X_test, Y_test


def get_result_for_NN(NN_function, X, Y):
    '''

    :param NN_function:
    :return: y_hat array (12*n_assets,)
    :return: rmse_test number 1
    '''

    function_name = function_dict[NN_function]


    ########################################
    # CV for validation
    # allocate space to store validation score and parameter
    val_dict = {}
    for param in param_grid['alpha_nn']:
        # train model with train dataset
        model = NN_function(input_shape=input_shape, alpha_nn=param)
        model.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size)
        Y_hat_val = model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(Y_hat_val, Y_val))
        val_dict[param] = rmse_val

    # fit best parameter
    best_param = min(val_dict)
    model_best = NN_function(input_shape=input_shape, alpha_nn=best_param)
    history = model_best.fit(X_train_val, Y_train_val, epochs=nb_epoch, batch_size=batch_size)
    print('Model parameters:\n', model_best.get_weights())
    
    print('Last layer bias:\n',model_best.get_weights()[-1])
    print('Last layer weight:\n',model_best.get_weights()[-2])
    # # summarize validation results
    # se = pd.Series(val_dict)
    # se.name = function_name
    # # save validation results
    # location = "./GSCV_" + function_name + "_" + filename + ".csv"
    # se.to_csv(location)
    #######################################################
    # plot the training curve
    histroy_loss_series = pd.Series(np.log(history.history['loss']))
    histroy_loss_series = histroy_loss_series.rename('loss')
    histroy_mse_series = pd.Series(history.history['mean_squared_error'])
    histroy_mse_series = histroy_mse_series.rename('mse')
    #######################################################

    # result
    feature_list = ['Asset{}'.format(i) for i in range(1, 56)]
    Y_hat = pd.DataFrame(index=range(12), columns=feature_list)

    for asset in range(1, 56):

        X_test, Y_test = get_testing_data(test_year, X, Y, asset)
        Y_hat['Asset{}'.format(asset)] = model_best.predict(X_test)

    # plot loss curve
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    histroy_mse_series.plot()
    plt.title('model MSE ' + function_name)
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend()
    plt.subplot(2, 1, 2)
    # summarize history for loss
    histroy_loss_series.plot()
    plt.title('model loss ' + function_name)
    plt.ylabel('ln(loss)')
    plt.xlabel('epoch')
    location = "./history_" + function_name + "_" + filename + ".pdf"
    plt.savefig(location)

    return Y_hat

def single_run(test_year, X, Y, file_name):
    '''
    :param test_year: 12 months in test year would be used for test
    :param file_name: output evrything with filename
    :return:
    '''
    print("=" * 20, test_year, "=" * 20)
    ##############################################
    # global variable
    global function_dict, input_shape, filename, Y_train_val, Y_val, Y_train, Y_test, X_train, X_val, X_train_val, X_test, test_start

    ##############################################
    #  give data after globalization
    filename = file_name

    # get training data
    X_train, X_train_val, X_val, Y_train, Y_train_val, Y_val = get_training_data(test_year, X, Y)
    print('X_train_val shape:{}\nY_train_val shape:{}\n'.format(X_train_val.shape, Y_train_val.shape))
    print('X_train shape:{}\nY_train shape:{}\n'.format(X_train.shape, Y_train.shape))

    # input_shape for Neural Network
    input_shape = (X.shape[1],)  # input_shape = (n_featrues, )

    ################################################################################################
    # get result from all NN
    # get name of each NN function
    ######## function dict would not run save according to the order ########
    # %%
    function_dict = {tanh_16_2_linear: 'tanh_16_2_linear',
                     tanh_16_2_relu: 'tanh_16_2_relu',         
                     tanh_16_4_2_linear: 'tanh_16_4_2_linear',
                     tanh_16_4_2_relu: 'tanh_16_4_2_relu',
                     tanh_16_8_4_2_linear: 'tanh_16_8_4_2_linear',
                     tanh_16_8_4_2_relu: 'tanh_16_8_4_2_relu'
                     }
    # allocate space
    Y_pred_df = []
    for i in range(1, 56):
        Y_pred_df.append(pd.DataFrame(index=range(12), columns=col, data=0))

    for func in col:
        function_name = function_dict[func]
        print("\n", "=" * 20 + "running ", function_name, "=" * 20)
        Y_hat = get_result_for_NN(func, X, Y)
        for asset in range(1,56):
            Y_pred_df[asset-1][function_name] = Y_hat['Asset{}'.format(asset)]

    return Y_pred_df
