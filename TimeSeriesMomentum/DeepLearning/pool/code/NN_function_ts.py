######################################
# This file contains 3 DL methods
# 3 relu 
######################################
from tuning import optimizer, lossfun, initializer
# fix random seed
from numpy.random import seed
seed(2019)
from tensorflow import set_random_seed
set_random_seed(2019)


from keras.layers.core import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras import regularizers
from keras import backend as K


###########################################
# 1. tanh_16_2
def tanh_16_2_linear(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)

    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)

    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)

    # hidden layer 3 #4 f layer
    layer3 = Dense(2, kernel_initializer=initializer, name='f', activation='tanh'
                   , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)

    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='linear')
    DL_output_raw = output(DL_hidden3)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])

    return simple_model

def tanh_16_2_relu(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)
    
    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)
                   
    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)
   
    # hidden layer 3 #4 f layer
    layer3 = Dense(2, kernel_initializer=initializer, name='f', activation='tanh'
                  , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)
   
    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='relu')
    DL_output_raw = output(DL_hidden3)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])
                   
    return simple_model

# 2. tanh_16_4_2
def tanh_16_4_2_linear(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)

    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)

    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)

    # hidden layer 3 # 8 f layer
    layer3 = Dense(4, kernel_initializer=initializer, activation='tanh'
                   , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)

    # hidden layer 4 Dropout
    layer4 = Dropout(0.8)
    DL_hidden4 = layer4(DL_hidden3)

    # hidden layer 5 #4 f layer
    layer5 = Dense(2, kernel_initializer=initializer, name='f', activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden5 = layer5(DL_hidden4)

    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='linear')
    DL_output_raw = output(DL_hidden5)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])

    return simple_model


def tanh_16_4_2_relu(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)
    
    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)
   
    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)
   
    # hidden layer 3 # 8 f layer
    layer3 = Dense(4, kernel_initializer=initializer, activation='tanh'
                  , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)
   
    # hidden layer 4 Dropout
    layer4 = Dropout(0.8)
    DL_hidden4 = layer4(DL_hidden3)
   
    # hidden layer 5 #4 f layer
    layer5 = Dense(2, kernel_initializer=initializer, name='f', activation='tanh',
                  kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden5 = layer5(DL_hidden4)
   
    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='relu')
    DL_output_raw = output(DL_hidden5)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])
                   
    return simple_model

# 3. tanh_16_8_4_2
def tanh_16_8_4_2_linear(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)

    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)

    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)

    # hidden layer 3 # 8 f layer
    layer3 = Dense(8, kernel_initializer=initializer, activation='tanh'
                   , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)

    # hidden layer 4 Dropout
    layer4 = Dropout(0.8)
    DL_hidden4 = layer4(DL_hidden3)

    # hidden layer 5 #4 f layer
    layer5 = Dense(4, kernel_initializer=initializer, name='f', activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden5 = layer5(DL_hidden4)
    
    # hidden layer 6 #4 f layer
    layer6 = Dense(2, kernel_initializer=initializer,  activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden6 = layer6(DL_hidden5)

    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='linear')
    DL_output_raw = output(DL_hidden6)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])

    return simple_model

def tanh_16_8_4_2_relu(input_shape, alpha_nn=0.01):
    # the input_big layer, DL_visible_0
    DL_visible_0 = Input(shape=input_shape)
    
    # hidden layer 1 #64
    layer1 = Dense(16, kernel_initializer=initializer, activation='tanh',
                   kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden1 = layer1(DL_visible_0)
   
    # hidden layer 2 Dropout
    layer2 = Dropout(0.8)
    DL_hidden2 = layer2(DL_hidden1)
   
    # hidden layer 3 # 8 f layer
    layer3 = Dense(8, kernel_initializer=initializer, activation='tanh'
                  , kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden3 = layer3(DL_hidden2)
   
    # hidden layer 4 Dropout
    layer4 = Dropout(0.8)
    DL_hidden4 = layer4(DL_hidden3)
   
    # hidden layer 5 #4 f layer
    layer5 = Dense(4, kernel_initializer=initializer, name='f', activation='tanh',
                  kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden5 = layer5(DL_hidden4)
   
    # hidden layer 6 #4 f layer
    layer6 = Dense(2, kernel_initializer=initializer,  activation='tanh',
                  kernel_regularizer=regularizers.l2(alpha_nn))
    DL_hidden6 = layer6(DL_hidden5)
   
    # hidden layer 5, the output layer, output dimension is M, add l2 regularization, y =
    # W *[f,xz] + b, no regularization added on the output layer
    #############
    output = Dense(1, kernel_initializer=initializer, activation='relu')
    DL_output_raw = output(DL_hidden6)
    simple_model = Model(inputs=DL_visible_0, outputs=DL_output_raw)
    simple_model.compile(loss=lossfun, optimizer=optimizer, metrics=['mse'])
                   
    return simple_model
