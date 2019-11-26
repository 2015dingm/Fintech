from keras import losses
from keras import optimizers, initializers
import numpy as np

# the purpose of tuning.py is to save tuning parameter for all NN
# which means all NN use same tuning from here
############################################
# parameter for building the model
optimizer = optimizers.RMSprop(lr=1e-4)
lossfun = losses.mean_squared_error
initializer = initializers.random_normal(stddev=0.01)

############################################
# parameter for fitting the autoencoder model
nb_epoch = 100
batch_size = 120

############################################
# parameter for cross_validation
# define the grid search parameters

############################################
# parameter for cross_validation
# define the grid search parameters
param_grid = {
    "alpha_nn": [1e-4] #np.exp(np.arange(-10, 5, step=1))
}

param_alpha = np.exp(np.arange(-10, 0, step=0.5))
l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

