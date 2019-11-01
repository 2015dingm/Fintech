from keras import losses
from keras import optimizers, initializers
import numpy as np

# the purpose of tuning.py is to save tuning parameter for all NN
# which means all NN use same tuning from here
############################################
# parameter for building the model
optimizer = optimizers.RMSprop(lr=1e-3)
lossfun = losses.mean_squared_error
initializer = initializers.random_normal(stddev=0.1)

############################################
# parameter for fitting the autoencoder model
nb_epoch = 500
batch_size = 120

############################################
# parameter for cross_validation
# define the grid search parameters
# 1. for Lasso, Ridge & Elastic Net
param_alpha = np.exp(np.arange(-20, 5, step=1))
l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

# 2. for gradient boost
GBtree_learning_rate = [0.1, 0.01]
GBtree_max_depth = [3, 5]
GBtree_n_estimators = [200]

# 3. for Random Forrest
RF_n_estimators = [100, 200, 500]
RF_max_depth = [3, 5]

