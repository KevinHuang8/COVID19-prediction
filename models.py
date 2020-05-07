import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.layers import Activation, Dropout, Input
from tensorflow.keras.regularizers import L1L2
import tensorflow.keras.backend as K

def pinball_loss(q, pred, true):
    e = true - pred
    return K.mean(K.maximum(q*e, (q - 1)*e))

def quantile_head(inputs, timesteps, target_features, reg=(0.1, 0.1)):
    if timesteps == 1:
        x = Dense(target_features, kernel_regularizer=L1L2(*reg))(inputs)
        return x
    else:
        x = TimeDistributed(Dense(target_features, 
            kernel_regularizer=L1L2(*reg)))(inputs)
        return x

def standard_LSTM(X_train, y_train, quantiles=False, loss='mse', optimizer='adam',
    units=64, layers=1, dropout=0, reg=(0.1, 0.1)):

    timesteps = y_train.shape[1]
    target_features = y_train.shape[2]

    if timesteps == 1:
        return_seq = False
    else:
        return_seq = True

    inputs = Input(shape=X_train.shape[1:])
    x = inputs
    for i in range(layers):
        x = LSTM(units, input_shape=X_train.shape[1:], 
            return_sequences=return_seq, activation='relu', 
            recurrent_regularizer=L1L2(0, 0), kernel_regularizer=L1L2(0, 0))(x)

    if dropout:
        x = Dropout(dropout)(x)

    if not quantiles:
        x = quantile_head(x, timesteps, target_features, reg)
        model = Model(inputs=inputs, outputs=x)
    else:
        qheads = []
        for q in range(quantiles):
            qheads.append(quantile_head(x, timesteps, target_features, reg))

        model = Model(inputs=inputs, outputs=qheads)

    if quantiles:
        loss = [lambda pred, true, q=q: pinball_loss(q, pred, true) for q
        in np.linspace(0.1, 1, quantiles, endpoint=False)]

    model.compile(loss=loss, optimizer=optimizer)

    return model