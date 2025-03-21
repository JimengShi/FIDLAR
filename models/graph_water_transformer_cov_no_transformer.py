#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: graph_transformer_parallel.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/20/23 15:31
"""

from pandas import DataFrame
from pandas import concat
from pandas import concat, read_csv
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Reshape, MaxPooling1D, Dense, Dropout, Concatenate, MultiHeadAttention, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv
from tensorflow.keras import backend as KB



def cnn_encoder(inputs, cnn_unit, l1_reg, l2_reg, dropout=0):
    x = layers.Conv1D(filters=cnn_unit, 
                      kernel_size=2, 
                      activation="relu",
                      padding='same', 
                      kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                     )(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], 
                      kernel_size=2,
                      activation="relu",
                      padding='same', 
                      kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                     )(x)
    
    return x



def graph_water_transformer_cov_gate_predictor_no_transformer(input_shape, gcn1, gcn2, lstm_unit, cnn_unit1, l1_reg, l2_reg, dropout, masked_value, gate_min, gate_max):
    """
    transformer_cov
    graph_water
    """
    # ======================== covariates with transformer ========================
    cov_inputs = keras.Input(shape=(input_shape), name='cov_inputs')
    
    cov = layers.Masking(mask_value=masked_value)(cov_inputs)
    cov = cnn_encoder(cov, cnn_unit1, l1_reg, l2_reg, dropout)
    #cov = layers.MaxPooling1D(pool_size=2, padding='same', name='pooling')(cov)
    cov = Dense(5)(cov)
    cov_reshape = layers.Reshape((5, input_shape[0]), name='cov_reshape')(cov)

    
    # ======================== water levels with GNN ========================
    inp_lap = Input((5, 5), name='inp_lap')
    inp_seq = Input((5, 72), name='inp_seq')

    # GCN
    x = GCNConv(gcn1, activation='relu', name='GCNConv1')([inp_seq, inp_lap])
    x = GCNConv(gcn2, activation='relu', name='GCNConv2')([x, inp_lap])


    # RNN
    xx = LSTM(lstm_unit, activation='relu', return_sequences=True, name='LSTM')(inp_seq)

    # ======================== CONCAT and Attention ========================
    x = Concatenate(name='concate')([cov_reshape, x, xx])
    x = layers.Attention(name='attention')([x, x])
    

    # ======================== final output ========================
    x = Flatten()(x)
    x = Dense(24*16, activation='relu')(x)
    x = layers.Reshape((24, 16))(x)
    #x = Dense(7, activation=bounded_activation(gate_min, gate_max))(x)
    x = Dense(7)(x)
    out = KB.minimum(KB.maximum(x, gate_min), gate_max)

    model = Model(inputs=[cov_inputs, inp_seq, inp_lap], outputs=out)
    
    return model, GCNConv
