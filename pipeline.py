import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Activation, Dropout
from tensorflow.keras.regularizers import L1L2

def standard_LSTM(X_train, y_train, loss='mse', optimizer='adam',
    units=64, dropout=0, reg=(0.1, 0.1)):

    timesteps = y_train.shape[1]
    target_features = y_train.shape[2]

    model = Sequential()

    if timesteps == 1:
        return_seq = False
    else:
        return_seq = True

    model.add(LSTM(64, input_shape=X_train.shape[1:], 
        return_sequences=return_seq, activation='relu',
        recurrent_regularizer=L1L2(*reg), kernel_regularizer=L1L2(*reg)))

    if dropout:
        model.add(Dropout(dropout))

    if timesteps == 1:
        model.add(Dense(target_features, kernel_regularizer=L1L2(*reg)))
    else:
        model.add(TimeDistributed(Dense(target_features, 
            kernel_regularizer=L1L2(*reg))))

    model.compile(loss='mse', optimizer='adam')

    return model

class Pipeline:
    def __init__(self, data_format, model, training):