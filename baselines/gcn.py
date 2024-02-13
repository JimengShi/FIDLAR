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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, BatchNormalization, Flatten, Reshape
from tensorflow.keras.models import load_model
from spektral.layers import GCNConv

from tensorflow.keras import backend as KB


# Custom activation function
def bounded_activation(min_value, max_value):
    def activation(x):
        return KB.minimum(KB.maximum(x, min_value), max_value)
    return activation



def gcn(n_nodes, n_timesteps, gcn1, gcn2, lstm_unit, dropout, masked_value):
    inp_lap = Input((n_nodes, n_nodes))
    inp_seq = Input((n_nodes, n_timesteps))
    #inp_seq = layers.Masking(mask_value=masked_value)(inp_seq)
    
    # GCN
    x = GCNConv(gcn1, activation='relu')([inp_seq, inp_lap])
    x = GCNConv(gcn2, activation='relu')([x, inp_lap])

    # RNN
    xx = LSTM(lstm_unit, activation='relu', return_sequences=True)(inp_seq)


    x = Concatenate()([x, xx])
    x = Flatten()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(96)(x)

    model = Model(inputs=[inp_seq, inp_lap], outputs=outputs)

    return model, GCNConv



def gcn_gate_generator(n_nodes, n_timesteps, gcn1, gcn2, lstm_unit, dropout, masked_value):
    inp_lap = Input((n_nodes, n_nodes))
    inp_seq = Input((n_nodes, n_timesteps))
    #inp_seq = layers.Masking(mask_value=masked_value)(inp_seq)
    
    # GCN
    x = GCNConv(gcn1, activation='relu')([inp_seq, inp_lap])
    x = GCNConv(gcn2, activation='relu')([x, inp_lap])

    # RNN
    xx = LSTM(lstm_unit, activation='relu', return_sequences=True)(inp_seq)


    x = Concatenate()([x, xx])
    x = Flatten()(x)
    x = Dense(24*7)(x)
    outputs = Reshape((24, 7))(x)

    model = Model(inputs=[inp_seq, inp_lap], outputs=outputs)

    return model, GCNConv




def gcn_gate_generator_1(n_nodes, n_timesteps, gcn1, gcn2, lstm_unit, dropout, masked_value, gate_min, gate_max):
    inp_lap = Input((n_nodes, n_nodes))
    inp_seq = Input((n_nodes, n_timesteps))
    #inp_seq = layers.Masking(mask_value=masked_value)(inp_seq)
    
    # GCN
    x = GCNConv(gcn1, activation='relu')([inp_seq, inp_lap])
    x = GCNConv(gcn2, activation='relu')([x, inp_lap])

    # RNN
    xx = LSTM(lstm_unit, activation='relu', return_sequences=True)(inp_seq)


    x = Concatenate()([x, xx])
    x = Flatten()(x)
    x = Dense(24*10)(x)
    x = Dense(24*7)(x)
    x = Reshape((24, 7))(x)
    out = KB.minimum(KB.maximum(x, gate_min), gate_max)
    #outputs = Reshape((24, 7))(out)

    model = Model(inputs=[inp_seq, inp_lap], outputs=out)

    return model, GCNConv