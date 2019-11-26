'''
    This is for one asset- Deep learning model
'''
from script import teststart_list, feature_num_list, test_year
#%%
from NN_function_ts import *
from tuning import batch_size, param_grid, param_alpha, l1_ratio
from tuning import nb_epoch_SUR, nb_epoch_pool, nb_epoch_timeseries, alpha_fixed_sur, alpha_fixed_pool, alpha_fixed_timeseries, verbose

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
import pickle



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


def get_result_for_NN_pool(function_name, function_dict, X, Y):
    '''

    :param NN_function:
    :return: y_hat array (12*n_assets,)
    :return: rmse_test number 1
    '''

    # function_name = function_dict[NN_function]
    NN_function = function_dict[function_name][0]


    ########################################
    # CV for validation
    # allocate space to store validation score and parameter
    val_dict = {}
    for param in param_grid['alpha_nn']:
        # train model with train dataset
        model = NN_function(input_shape=input_shape, alpha_nn=param)
        model.fit(X_train_pool, Y_train_pool, epochs=nb_epoch_pool, batch_size=batch_size)
        Y_hat_val = model.predict(X_val_pool)
        rmse_val = np.sqrt(mean_squared_error(Y_hat_val, Y_val_pool))
        val_dict[param] = rmse_val

    # fit best parameter
    best_param = min(val_dict)
    model_best = NN_function(input_shape=input_shape, alpha_nn=best_param)
    history = model_best.fit(X_train_val_pool, Y_train_val_pool, epochs=nb_epoch_pool, batch_size=batch_size)
    print('NN function:', NN_function)
    print('Model parameters:\n', model_best.get_weights())

    print('Last layer bias:\n', model_best.get_weights()[-1])
    print('Last layer weight:\n', model_best.get_weights()[-2])
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

    weights = list()
    biases = list()
    # NOTE! THIS PART WOULD BE DIFFERENT FOR DIFFERENT NN STRUCTURE
    if function_name == 'NN_16_4_2_linear' or function_name == 'NN_16_4_2_relu':
        layer_col = [1, 3, 5]
    elif function_name == 'NN_16_8_4_2_linear' or function_name == 'NN_16_8_4_2_relu':
        layer_col = [1, 3, 5, 7]
    else:
        layer_col = [1, 3]

    for i in layer_col:
        layer = model_best.layers[i]
        # get_weights()[0] is weight matrix
        weights.append(layer.get_weights()[0])
        # get_weights()[1] is biases vector
        biases.append(layer.get_weights()[1])

    # with open(filename + function_name + '_pool_yhat.pickle', 'wb') as f:
    # pickle.dump(y_hat_list, f)

    with open(filename + function_name + '_pool_weights.pickle', 'wb') as f:
        pickle.dump(weights, f)

    with open(filename + function_name + '_pool_biases.pickle', 'wb') as f:
        pickle.dump(biases, f)

    return Y_hat


def get_result_for_NN_SUR(function_name, function_dict, asset, batchsize):
    '''

    :param NN_function:
    :return: y_hat array (12*n_asset,)
    :return: rmse_test number 1
    '''

    # loadname = filename.replace(' ', '')[:-3] + 'pool'

    f = open(filename + function_name + '_pool_weights.pickle', 'rb')
    weights = pickle.load(f)
    f.close()

    f = open(filename + function_name + '_pool_biases.pickle', 'rb')
    biases = pickle.load(f)
    f.close()
    ###############################################################
    # allocate space
    y_hat_list = []


    ###############################################################################################
    # 3. DL-sur (n_asset model )
    #############################
    yhat_array_list = []
    d = {}
    plt.figure(figsize=(10, 10))# n_asset industry in one plot
    # for i in range(0, n_asset):
    # get NN function from function_dict
    # * regular means the normal initialization and loss
    # * dl_sur has inherited initialization and loss
    # regular = function_dict[function_name][0]
    dl_sur = function_dict[function_name][1]

    # # Gridsearch CV for validation
    # estimator = KerasRegressor(build_fn=dl_sur, input_shape=input_big_shape,
    #                            nb_epoch=nb_epoch_SUR, batch_size=batch_size, verbose=1)

    # # cross validation
    # # setting customized CV splitter for 1 validation on validation dataset
    # # set all index to 0, 0 is the index of validation dataset of 1st round
    # test_fold = np.zeros(X_train_val_big1[:, :, i].shape[0])
    # # set index of training data to -1: never be set into validation dataset
    # test_fold[:X_train_big1[:, :, i].shape[0]] = -1
    # ps = PredefinedSplit(test_fold=test_fold)

    # # grid search parameter
    # weights_param = [weights]  # inherited weights
    # biases_param = [biases]  # inherited biases
    # alpha_nn_param = [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4]  # tuning for DL-sur

    # # though param_grid have three args, CV is made based on alpha_nn_param, as weights and biases are fixed.
    # param_grid = dict(weights=weights_param,
    #                   biases=biases_param, alpha_nn=alpha_nn_param)
    # # refit = True: The refitted estimator is made available at the best_estimator_ attribute and
    # # permits using predict directly on this GridSearchCV instance.
    # modeltemp = GridSearchCV(estimator=estimator, param_grid=param_grid,
    #                          scoring='neg_mean_squared_error', cv=ps, verbose=1, return_train_score=True,
    #                          refit=True, n_jobs=-1)

    # modeltemp.fit(X_train_val_big1[:, :, i], Y_train_val[:, i])
    # best = modeltemp.best_params_

    # # save validation results
    # df = pd.DataFrame(modeltemp.cv_results_)
    # location = "./GSCV_" + function_name + \
    #     "_" + filename + "_asset"+str(i) + ".csv"
    # df[['mean_test_score', 'std_test_score', 'params']].to_csv(location)


    best_param = alpha_fixed_sur
    model_best = dl_sur(input_shape=input_shape, weights=weights, biases=biases, alpha_nn=best_param)
    history = model_best.fit(X_train_val_sur, Y_train_val_sur, epochs=nb_epoch_SUR, batch_size=batchsize,verbose=verbose)

    # loss curve of each asset
    se = pd.Series(np.log(history.history['loss'])).rename('SUR_' + 'asset' + str(asset))
    se.plot(legend=True)

    plt.title('model LOSS')
    plt.ylabel('ln(LOSS)')
    plt.xlabel('epoch')
    location = "./"+filename +'_SUR_' + function_name + '_asset' + str(asset) + ".pdf"
    plt.savefig(location)

    # result
    Y_hat = model_best.predict(X_test_sur)
    # y_hat = np.array(model_best.predict(X_test_sur))
    # yhat_array_list.append(y_hat.ravel())
    # key = 'industry' + str(asset)
    # d[key] = np.sqrt(mean_squared_error(Y_test_sur, y_hat))
    #
    # yhat_array = np.stack(yhat_array_list, axis=1)
    # y_hat_list.append(yhat_array)

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
    global Y_train_val_pool, Y_val_pool, Y_train_pool, X_train_pool, X_val_pool, X_train_val_pool
    global X_train_sur, X_val_sur, X_train_val_sur, X_test_sur, Y_train_sur, Y_val_sur, Y_train_val_sur, Y_test_sur

    ##############################################
    #  give data after globalization
    filename = file_name

    # get training data for pool
    X_train_pool, X_train_val_pool, X_val_pool, Y_train_pool, Y_train_val_pool, Y_val_pool = get_training_data(test_year, X, Y)

    print('X_train_val_pool shape:{}\nY_train_val shape:{}\n'.format(X_train_val_pool.shape, Y_train_val_pool.shape))
    print('X_train_pool shape:{}\nY_train shape:{}\n'.format(X_train_pool.shape, Y_train_pool.shape))

    # input_shape for Neural Network
    input_shape = (X.shape[1],)  # input_shape = (n_featrues, )

    ################################################################################################
    # get result from all NN
    # get name of each NN function
    ######## function dict would not run save according to the order ########
    # %%
    function_dict = {'NN_16_2_linear': [tanh_16_2_linear, SUR_16_2_linear],
                     'NN_16_2_relu': [tanh_16_2_relu, SUR_16_2_relu],
                     'NN_16_4_2_linear': [tanh_16_4_2_linear, SUR_16_4_2_linear],
                     'NN_16_4_2_relu': [tanh_16_4_2_relu, SUR_16_4_2_relu],
                     'NN_16_8_4_2_linear': [tanh_16_8_4_2_linear, SUR_16_8_4_2_linear],
                     'NN_16_8_4_2_relu': [tanh_16_8_4_2_relu, SUR_16_8_4_2_relu]
                     }
   
    # allocate space
    col = ['NN_16_2_linear', 'NN_16_2_relu', 'NN_16_4_2_linear', 'NN_16_4_2_relu', 'NN_16_8_4_2_linear', 'NN_16_8_4_2_relu']
    Y_pred_df = []
    for i in range(1, 56):
        Y_pred_df.append(pd.DataFrame(index=range(12), columns=col, data=0))


    for func in ['NN_16_2_linear', 'NN_16_2_relu', 'NN_16_4_2_linear', 'NN_16_4_2_relu', 'NN_16_8_4_2_linear', 'NN_16_8_4_2_relu']:
        Y_hat = get_result_for_NN_pool(func, function_dict, X, Y)
        # training for sur
        for asset in range(1, 56):
            feature_num = 0
            for i in range(asset):
                feature_num = feature_num + feature_num_list[i]
            feature_start = feature_num - feature_num_list[asset - 1]
            feature_end = feature_num

            X_sub = X.iloc[feature_start:feature_end, :].values
            Y_sub = Y[feature_start:feature_end].values

            test_start = teststart_list[asset - 1]

            X_train_sur, X_val_sur, X_train_val_sur, X_test_sur, Y_train_sur, Y_val_sur, Y_train_val_sur, Y_test_sur = get_data_one_asset(test_year, test_start, X_sub, Y_sub)

            print("\n", "="*20+"running ", func, ", Asset", asset, "="*20)

            Y_hat = get_result_for_NN_SUR(func, function_dict, asset, batch_size)
            Y_pred_df[asset-1][func] = Y_hat
            # for asset in range(1,56):
            #     Y_pred_df[asset-1][function_name] = Y_hat['Asset{}'.format(asset)]

    return Y_pred_df
