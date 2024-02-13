#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: graph_transformer_series.py
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


def transformer_encoder(inputs, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout=0):
    """
    transformer encoder
    """
    # Normalization and Attention
    x = layers.MultiHeadAttention(key_dim=head_size, 
                                  num_heads=num_heads, 
                                  dropout=dropout, 
                                  kernel_regularizer=l2(atte_reg)
                                 )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x1 = layers.LayerNormalization(epsilon=1e-6)(res)
    #res = x + inputs

#     # Feed Forward Part
#     x = layers.Conv1D(filters=ff_dim, 
#                       kernel_size=2, 
#                       activation="relu",
#                       padding='same', 
#                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
#                      )(x1)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], 
#                       kernel_size=2,
# #                       activation="relu",
#                       padding='same', 
#                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
#                      )(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward Part
    x = layers.Dense(units=ff_dim, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units=inputs.shape[-1], activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    return x + res



def graph_global_transformer_local(gcn1, gcn2, gcn3, lstm_unit, num_transformer_blocks, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout, masked_value, embed_feat, S26_shape, S25B_shape, S25A_shape, S1_shape, S4_shape):
    # S26
    S26_inputs = keras.Input(shape=S26_shape)
    S26 = layers.Masking(mask_value=masked_value)(S26_inputs)
    for _ in range(num_transformer_blocks):
        S26 = transformer_encoder(S26, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S26 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S26)
    S26 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                        kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                       )(S26)


    # S25B
    S25B_inputs = keras.Input(shape=S25B_shape)
    S25B = layers.Masking(mask_value=masked_value)(S25B_inputs)
    for _ in range(num_transformer_blocks):
        S25B = transformer_encoder(S25B, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S25B = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S25B)
    S25B = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                         kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                        )(S25B)


    # S25A
    S25A_inputs = keras.Input(shape=S25A_shape)
    S25A = layers.Masking(mask_value=masked_value)(S25A_inputs)
    for _ in range(num_transformer_blocks):
        S25A = transformer_encoder(S25A, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S25A = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S25A)
    S25A = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                         kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                        )(S25A)


    # S1
    S1_inputs = keras.Input(shape=S1_shape)
    S1 = layers.Masking(mask_value=masked_value)(S1_inputs)
    for _ in range(num_transformer_blocks):
        S1 = transformer_encoder(S1, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S1 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S1)
    S1 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                      )(S1)

    # S4
    S4_inputs = keras.Input(shape=S4_shape)
    S4 = layers.Masking(mask_value=masked_value)(S4_inputs)
    for _ in range(num_transformer_blocks):
        S4 = transformer_encoder(S4, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S4 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S4)
    S4 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                      )(S4)

    # concat
    all_concat = layers.concatenate([S26, S25B, S25A, S1, S4], name='concat_local')    # (batch_size, 48, 15)
    all_concat_reshape = layers.Reshape((5, S4_shape[0]*embed_feat), name='reshape_global')(all_concat)


    # GNN
    inp_lap = Input((5, 5), name='input_lap')
    x = GCNConv(gcn1, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                name='GCNConv1'
               )([all_concat_reshape, inp_lap])
    x = GCNConv(gcn2, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                name='GCNConv2'
               )([x, inp_lap])


    # RNN
    xx = LSTM(lstm_unit, 
              activation='relu', 
              return_sequences=True, 
              name='LSTM')(all_concat_reshape)


    x = Concatenate(name='concat_global')([x, xx])
    
#     x = Dense(32)(x) 
    
    # Attention between cov and water level
    x = layers.Attention(name='attention')([x, x])


    x = Flatten(name='Flatten')(x)
    x = Dense(96, name='final_dense')(x) 


    model = keras.models.Model([S26_inputs, S25B_inputs, S25A_inputs, S1_inputs, S4_inputs, inp_lap], x)
    # model.summary()


    return model, GCNConv




def graph_global_transformer_local_for_gate_predictor(gcn1, gcn2, lstm_unit, num_transformer_blocks, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout, masked_value, embed_feat, S26_shape, S25B_shape, S25A_shape, S1_shape, S4_shape):
    # S26
    S26_inputs = keras.Input(shape=S26_shape)
    S26 = layers.Masking(mask_value=masked_value)(S26_inputs)
    for _ in range(num_transformer_blocks):
        S26 = transformer_encoder(S26, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S26 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S26)
    S26 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                        kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                       )(S26)


    # S25B
    S25B_inputs = keras.Input(shape=S25B_shape)
    S25B = layers.Masking(mask_value=masked_value)(S25B_inputs)
    for _ in range(num_transformer_blocks):
        S25B = transformer_encoder(S25B, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S25B = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S25B)
    S25B = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                         kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                        )(S25B)


    # S25A
    S25A_inputs = keras.Input(shape=S25A_shape)
    S25A = layers.Masking(mask_value=masked_value)(S25A_inputs)
    for _ in range(num_transformer_blocks):
        S25A = transformer_encoder(S25A, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S25A = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S25A)
    S25A = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                         kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                        )(S25A)


    # S1
    S1_inputs = keras.Input(shape=S1_shape)
    S1 = layers.Masking(mask_value=masked_value)(S1_inputs)
    for _ in range(num_transformer_blocks):
        S1 = transformer_encoder(S1, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S1 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S1)
    S1 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                      )(S1)

    # S4
    S4_inputs = keras.Input(shape=S4_shape)
    S4 = layers.Masking(mask_value=masked_value)(S4_inputs)
    for _ in range(num_transformer_blocks):
        S4 = transformer_encoder(S4, head_size, num_heads, ff_dim, atte_reg, l1_reg, l2_reg, dropout)
    # S4 = layers.MaxPooling1D(pool_size=pool_size, padding='same')(S4)
    S4 = layers.Conv1D(filters=embed_feat, kernel_size=2, activation="relu", padding='same', 
                       kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
                      )(S4)

    # concat
    all_concat = layers.concatenate([S26, S25B, S25A, S1, S4], name='concat_local')    # (batch_size, 48, 15)
    all_concat_reshape = layers.Reshape((5, S4_shape[0]*embed_feat), name='reshape_global')(all_concat)


    # GNN
    inp_lap = Input((5, 5), name='input_lap')
    x = GCNConv(gcn1, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                name='GCNConv1'
               )([all_concat_reshape, inp_lap])
    x = GCNConv(gcn2, 
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                name='GCNConv2'
               )([x, inp_lap])


    # RNN
    xx = LSTM(lstm_unit, 
              activation='relu', 
              return_sequences=True, 
              name='LSTM')(all_concat_reshape)


    x = Concatenate(name='concat_global')([x, xx])
    
    
    # Attention between cov and water level
    x = layers.Attention(name='attention')([x, x])


    x = Flatten(name='Flatten')(x)
    x = Dense(24*7)(x)
    x = Reshape((24, 7))(x)
    #x = Dense(96, name='final_dense')(x) 


    model = keras.models.Model([S26_inputs, S25B_inputs, S25A_inputs, S1_inputs, S4_inputs, inp_lap], x)
    # model.summary()


    return model, GCNConv
