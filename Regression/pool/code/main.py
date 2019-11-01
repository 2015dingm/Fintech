import os
import pandas as pd
import warnings
from single_run import *
warnings.filterwarnings("ignore", category=FutureWarning)

# fix random seed
from numpy.random import seed
seed(2019)
from tensorflow import set_random_seed
set_random_seed(2019)


col = ['AutoEncoder (1 components)','AutoEncoder (3 components)','PCA (1 components)','PCA (3 components)','PLS (1 components)','PLS (3 components)',
'OLS regression','Lasso regression','Ridge regression','Elastic Net','Boosted regression Tree','Random Forrest']

# =============================================================================
# Read data
# =============================================================================
ret=pd.read_excel('../../data/Return.xlsx','Return')
ret=ret[ret['YM']>=198501]# make sure all returns are after 198501
ret.loc[:,'RF']=ret['Rf']+1#capital RF is the gross risk free rate
#%%
ret.loc[:,'Return']=ret['ExRet']+ret['RF']
feature_list = ['Ret-{}'.format(i) for i in range(1,61) ]
#################################################################################




##############################################################################
# filename to seperate different spatch
filename = 'year{}'.format(test_year)

feature_sub_list = []
exret_sub_list = []
teststart_list = []
feature_num_list = []

for asset in range(1,56):
    ret_sub=ret[ret['ID']==asset]
    # -1, -2, ....., -60
    # -1
    ret_sub.loc[:,'Ret-1'] = (ret_sub['Return'])

    ret_sub.loc[:,'Ret-2'] = (ret_sub['Return'].shift(1))

    ret_sub.loc[:,'Ret-3'] = (ret_sub['Return'].shift(2))

    ret_sub.loc[:,'Ret-4'] = (ret_sub['Return'].shift(3))

    ret_sub.loc[:,'Ret-5'] = (ret_sub['Return'].shift(4))

    ret_sub.loc[:,'Ret-6'] = (ret_sub['Return'].shift(5))

    ret_sub.loc[:,'Ret-7'] = (ret_sub['Return'].shift(6))

    ret_sub.loc[:,'Ret-8'] = (ret_sub['Return'].shift(7))

    ret_sub.loc[:,'Ret-9'] = (ret_sub['Return'].shift(8))

    ret_sub.loc[:,'Ret-10'] = (ret_sub['Return'].shift(9))

    ret_sub.loc[:,'Ret-11'] = (ret_sub['Return'].shift(10))

    ret_sub.loc[:,'Ret-12'] = (ret_sub['Return'].shift(11))

    ret_sub.loc[:,'Ret-13'] = (ret_sub['Return'].shift(12))

    ret_sub.loc[:,'Ret-14'] = (ret_sub['Return'].shift(13))

    ret_sub.loc[:,'Ret-15'] = (ret_sub['Return'].shift(14))

    ret_sub.loc[:,'Ret-16'] = (ret_sub['Return'].shift(15))

    ret_sub.loc[:,'Ret-17'] = (ret_sub['Return'].shift(16))

    ret_sub.loc[:,'Ret-18'] = (ret_sub['Return'].shift(17))

    ret_sub.loc[:,'Ret-19'] = (ret_sub['Return'].shift(18))

    ret_sub.loc[:,'Ret-20'] = (ret_sub['Return'].shift(19))

    ret_sub.loc[:,'Ret-21'] = (ret_sub['Return'].shift(20))

    ret_sub.loc[:,'Ret-22'] = (ret_sub['Return'].shift(21))

    ret_sub.loc[:,'Ret-23'] = (ret_sub['Return'].shift(22))

    ret_sub.loc[:,'Ret-24'] = (ret_sub['Return'].shift(23))

    ret_sub.loc[:,'Ret-25'] = (ret_sub['Return'].shift(24))

    ret_sub.loc[:,'Ret-26'] = (ret_sub['Return'].shift(25))

    ret_sub.loc[:,'Ret-27'] = (ret_sub['Return'].shift(26))

    ret_sub.loc[:,'Ret-28'] = (ret_sub['Return'].shift(27))

    ret_sub.loc[:,'Ret-29'] = (ret_sub['Return'].shift(28))

    ret_sub.loc[:,'Ret-30'] = (ret_sub['Return'].shift(29))

    ret_sub.loc[:,'Ret-31'] = (ret_sub['Return'].shift(30))

    ret_sub.loc[:,'Ret-32'] = (ret_sub['Return'].shift(31))

    ret_sub.loc[:,'Ret-33'] = (ret_sub['Return'].shift(32))

    ret_sub.loc[:,'Ret-34'] = (ret_sub['Return'].shift(33))

    ret_sub.loc[:,'Ret-35'] = (ret_sub['Return'].shift(34))

    ret_sub.loc[:,'Ret-36'] = (ret_sub['Return'].shift(35))

    ret_sub.loc[:,'Ret-37'] = (ret_sub['Return'].shift(36))

    ret_sub.loc[:,'Ret-38'] = (ret_sub['Return'].shift(37))

    ret_sub.loc[:,'Ret-39'] = (ret_sub['Return'].shift(38))

    ret_sub.loc[:,'Ret-40'] = (ret_sub['Return'].shift(39))

    ret_sub.loc[:,'Ret-41'] = (ret_sub['Return'].shift(40))

    ret_sub.loc[:,'Ret-42'] = (ret_sub['Return'].shift(41))

    ret_sub.loc[:,'Ret-43'] = (ret_sub['Return'].shift(42))

    ret_sub.loc[:,'Ret-44'] = (ret_sub['Return'].shift(43))

    ret_sub.loc[:,'Ret-45'] = (ret_sub['Return'].shift(44))

    ret_sub.loc[:,'Ret-46'] = (ret_sub['Return'].shift(45))

    ret_sub.loc[:,'Ret-47'] = (ret_sub['Return'].shift(46))

    ret_sub.loc[:,'Ret-48'] = (ret_sub['Return'].shift(47))

    ret_sub.loc[:,'Ret-49'] = (ret_sub['Return'].shift(48))

    ret_sub.loc[:,'Ret-50'] = (ret_sub['Return'].shift(49))

    ret_sub.loc[:,'Ret-51'] = (ret_sub['Return'].shift(50))

    ret_sub.loc[:,'Ret-52'] = (ret_sub['Return'].shift(51))

    ret_sub.loc[:,'Ret-53'] = (ret_sub['Return'].shift(52))

    ret_sub.loc[:,'Ret-54'] = (ret_sub['Return'].shift(53))

    ret_sub.loc[:,'Ret-55'] = (ret_sub['Return'].shift(54))

    ret_sub.loc[:,'Ret-56'] = (ret_sub['Return'].shift(55))

    ret_sub.loc[:,'Ret-57'] = (ret_sub['Return'].shift(56))

    ret_sub.loc[:,'Ret-58'] = (ret_sub['Return'].shift(57))

    ret_sub.loc[:,'Ret-59'] = (ret_sub['Return'].shift(58))

    ret_sub.loc[:,'Ret-60'] = (ret_sub['Return'].shift(59))


    # return forward 1
    ret_sub.loc[:, 'ExRet_f1'] = ret_sub['ExRet'].shift(-1)
    ret_sub.dropna(inplace=True)
    feature_num = len(ret_sub)
    teststart = len(ret_sub[ret_sub['YM'] < 200512])  # note: it's <199912 not <=, so that the first month of test would be predicting 200001 using feature observed at 199912

    feature_sub_list.append(ret_sub[feature_list])
    exret_sub_list.append(ret_sub['ExRet_f1'])
    teststart_list.append(teststart)
    feature_num_list.append(feature_num)

feature = pd.concat(feature_sub_list, axis=0, ignore_index=True)
exret = pd.concat(exret_sub_list, axis=0, ignore_index=True)



if __name__ == '__main__':
    print(os.getcwd())
    Y_pred_df = single_run(test_year, feature, exret, filename)
    print(os.getcwd())
    # make a folder to store result
    folder = './result_new/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # save data
    for asset in range(1,56):
        Y_pred_df[asset-1].to_csv(os.path.join(folder, filename + '_asset' + str(asset) + '_yhat.csv'))






