'''
This is for pool model
'''
from script import col, teststart_list, feature_num_list
from tuning import nb_epoch, batch_size, param_alpha, l1_ratio, \
    GBtree_learning_rate, GBtree_max_depth, GBtree_n_estimators,RF_n_estimators, RF_max_depth
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

from keras.models import Model
from keras.layers import Dense, Input


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
    Y_train_list = []
    Y_train_val_list = []

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
        Y_train_list.append(Y_train_)
        Y_train_val_list.append(Y_train_val_)

        # if asset == asset_i:
        #     X_val = X_val_
        #     X_test = X_test_
        #     Y_val = Y_val_
        #     Y_test = Y_test_
        # print('X_train_val shape:{}\nY_train_val shape:{}\n'.format(X_train_val.shape, Y_train_val.shape))

    X_train = np.concatenate(X_train_list, axis=0)
    X_train_val = np.concatenate(X_train_val_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    Y_train_val = np.concatenate(Y_train_val_list, axis=0)

    return X_train, X_train_val, Y_train, Y_train_val

def get_testing_data(test_year, X, Y, asset):
    feature_num = 0
    for i in range(asset):
        feature_num = feature_num + feature_num_list[i]
    feature_start = feature_num - feature_num_list[asset - 1]
    feature_end = feature_num

    X_sub = X.iloc[feature_start:feature_end, :].values
    Y_sub = Y[feature_start:feature_end].values

    test_start = teststart_list[asset - 1]

    X_train_, X_val_, X_train_val_, X_test_, Y_train_, Y_val_, Y_train_val_, Y_test_ = get_data_one_asset(test_year,
                                                                                                          test_start,
                                                                                                          X_sub, Y_sub)

    X_val = X_val_
    X_test = X_test_
    Y_val = Y_val_
    Y_test = Y_test_

    return X_val, X_test, Y_val, Y_test

def autoencoder_model(X_train, encoding_dim = 3):
    '''
    :param encoding_dim: the dimension after dimension reduction
    :param X_train: training data X that need dimension reduction
    :return: encoder [the model that can implement dimension reduction from autoencoder]
    '''
    # input layer
    input_layer = Input(shape=(input_dim,))
    # encoder layers
    encoded = Dense(16, activation=None)(input_layer)
    encoder_output = Dense(encoding_dim)(encoded)
    #decoder layers
    decoded = Dense(16,activation=None)(encoder_output)
    decoder_output = Dense(input_dim, activation=None)(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_layer, output=decoder_output)
    encoder = Model(input=input_layer, output=encoder_output)
    # compile autoencoder
    autoencoder.compile(optimizer='adam',loss='mse')

    # training
    autoencoder.fit(X_train, X_train, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True)

    return encoder

def single_run(test_year, X, Y, file_name):
    '''
    :param test_year: 12 months in test year would be used for test
    :param file_name: output everything with filename
    :return:
    '''
    print("="*20,test_year,"="*20)
    ##############################################
    # global variable
    global input_dim, X_train, Y_train, X_train_val, Y_train_val, X_test, Y_test, filename

    ##############################################
    #  give data after globalization
    filename = file_name


    # get training data
    X_train, X_train_val, Y_train, Y_train_val = get_training_data(test_year, X, Y)
    print('X_train_val shape:{}\nY_train_val shape:{}\n'.format(X_train_val.shape, Y_train_val.shape))
    print('X_train shape:{}\nY_train shape:{}\n'.format(X_train.shape, Y_train.shape))


    # input_shape for Neural Network autoencoder
    input_dim = X.shape[1] # k is the number of feature

    # allocate space
    Y_pred_df = []
    for i in range(1,56):
        Y_pred_df.append(pd.DataFrame(index=range(12), columns=col, data=0))

    # 0. Generate customized cross validation
    test_fold = np.zeros(X_train_val.shape[0])
    # set index of training data to -1: never be set into validation dataset
    test_fold[:X_train.shape[0]] = -1
    ps = PredefinedSplit(test_fold=test_fold)

    ####################################################################
    # get result from all other methods
    # 0. autoencoder 1
    encoding_dim = 1
    encoder = autoencoder_model(X_train_val, encoding_dim=encoding_dim)
    X_train_reduced = encoder.predict(X_train_val)

    # ols
    regr = LinearRegression()
    regr.fit(X_train_reduced, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        X_test_reduced = encoder.predict(X_test)
        Y_pred = regr.predict(X_test_reduced)
        Y_pred_df[asset-1]['AutoEncoder (1 components)'] = Y_pred

    # 1. autoencoder 3
    encoding_dim = 3
    encoder = autoencoder_model(X_train_val, encoding_dim=encoding_dim)
    X_train_reduced = encoder.predict(X_train_val)

    # ols
    regr = LinearRegression()
    regr.fit(X_train_reduced, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        X_test_reduced = encoder.predict(X_test)
        Y_pred = regr.predict(X_test_reduced)
        Y_pred_df[asset-1]['AutoEncoder (3 components)'] = Y_pred

    # 2. pca = PCA(n_components= 1)
    # pca transform data
    num_component = 1
    pca = PCA(n_components=num_component)
    X_train_reduced = pca.fit_transform(X_train_val)
    # ols
    regr = LinearRegression()
    regr.fit(X_train_reduced, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        X_test_reduced = pca.transform(X_test)
        Y_pred = regr.predict(X_test_reduced)
        Y_pred_df[asset-1]['PCA (1 components)'] = Y_pred

    # 3. pca = PCA(n_components= 3)
    # pca transform data
    num_component = 3
    pca = PCA(n_components=num_component)
    X_train_reduced = pca.fit_transform(X_train_val)

    # ols
    regr = LinearRegression()
    regr.fit(X_train_reduced, Y_train_val)
    for asset in range(1, 56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        X_test_reduced = pca.transform(X_test)
        Y_pred = regr.predict(X_test_reduced)
        Y_pred_df[asset - 1]['PCA (3 components)'] = Y_pred

    # 4. pls = PLS(n_components= 1)
    # PLS transform data
    num_component = 1
    pls = PLSRegression(n_components=num_component)
    # pls fit data
    pls.fit(X_train_val, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = pls.predict(X_test)
        Y_pred_df[asset - 1]['PLS (1 components)'] = Y_pred

    # 5. pls = PLS(n_components= 3)
    # PLS transform data
    num_component = 3
    pls = PLSRegression(n_components=num_component)
    # pls fit data
    pls.fit(X_train_val, Y_train_val)
    for asset in range(1, 56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = pls.predict(X_test)
        Y_pred_df[asset - 1]['PLS (3 components)'] = Y_pred

    # 6. OLS regression
    regr = LinearRegression()
    # fit data
    regr.fit(X_train_val, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = regr.predict(X_test)
        Y_pred_df[asset-1]['OLS regression'] = Y_pred

    # 7. LassoCV
    #reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html
    reg = LassoCV(cv=ps, alphas=param_alpha, n_jobs=-1, selection='random')
    reg.fit(X_train_val, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = reg.predict(X_test)
        Y_pred_df[asset-1]['Lasso regression'] = Y_pred

    # 8. ridgeCV
    reg = ElasticNetCV(l1_ratio=0, alphas=param_alpha, cv=ps,verbose=0, n_jobs=-1, selection='random', max_iter=3000)
    reg.fit(X_train_val, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = reg.predict(X_test)
        Y_pred_df[asset-1]['Ridge regression'] = Y_pred

    # 9. Multitask Elastics CV
    reg = ElasticNetCV(l1_ratio=l1_ratio, alphas=param_alpha,cv=ps, verbose=0, n_jobs=-1, selection='random', max_iter=3000)
    reg.fit(X_train_val, Y_train_val)
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = reg.predict(X_test)
        Y_pred_df[asset-1]['Elastic Net'] = Y_pred

    # 10. Gradient boost tree regression CV
    # reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

    # Using grid search to CV
    # define estimator
    estimator = GradientBoostingRegressor()
    # define parameter grid
    param_grid = {'learning_rate': GBtree_learning_rate,
                  'max_depth': GBtree_max_depth,
                  'n_estimators': GBtree_n_estimators}
    best_reg = GridSearchCV(estimator=estimator, cv=ps, param_grid=param_grid, n_jobs=-1)

    y_train_val = Y_train_val
    best_reg.fit(X_train_val, y_train_val)
    print('CV estimator:{}\n'.format( best_reg.best_estimator_))
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = best_reg.predict(X_test)
        Y_pred_df[asset-1]['Boosted regression Tree'] = Y_pred

    # 11. Random Forrest regression CV

    # Using grid search to CV
    # define estimator
    estimator = RandomForestRegressor()
    # define parameter grid
    param_grid = {'n_estimators': RF_n_estimators,
                  'max_depth': RF_max_depth}
    best_reg = GridSearchCV(estimator=estimator, cv=ps, param_grid=param_grid, n_jobs=-1)

    best_reg.fit(X_train_val, Y_train_val)
    print('CV estimator:{}\n'.format( best_reg.best_estimator_))
    for asset in range(1,56):
        X_val, X_test, Y_val, Y_test = get_testing_data(test_year, X, Y, asset)
        # print('X_test shape:{}\nY_test shape:{}\n'.format(X_test.shape, Y_test.shape))
        Y_pred = best_reg.predict(X_test)
        Y_pred_df[asset-1]['Random Forrest'] = Y_pred

    return Y_pred_df


