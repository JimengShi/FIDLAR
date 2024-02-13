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
from tensorflow.keras.layers import LSTM, SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as KB



def rnn(input_shape, rnn_unit, l1_reg, l2_reg, dropout, masked_value):
    """
    l1_reg: 0
    l2_reg: 1e-5
    """
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = SimpleRNN(rnn_unit, 
                  activation='relu', 
                  kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  return_sequences=True)(masked_inputs)
    x = Dropout(dropout)(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(96)(x)

    rnn_model = Model(inputs=inputs, outputs=outputs)
    #rnn_model.summary()


    return rnn_model



def rnn_gate_generator(input_shape, rnn_unit, l1_reg, l2_reg, dropout, masked_value):
    """
    l1_reg: 0
    l2_reg: 1e-5
    """
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = SimpleRNN(rnn_unit, 
                  activation='relu', 
                  kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  return_sequences=True)(masked_inputs)
    x = Dropout(dropout)(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = layers.Dense(7)(x)
    
    rnn_model = Model(inputs=inputs, outputs=x)
    #rnn_model.summary()


    return rnn_model



def rnn_gate_generator_1(input_shape, rnn_unit, l1_reg, l2_reg, dropout, masked_value, gate_min, gate_max):
    """
    l1_reg: 0
    l2_reg: 1e-5
    """
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = SimpleRNN(rnn_unit, 
                  activation='relu', 
                  kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  return_sequences=True)(masked_inputs)
    x = Dropout(dropout)(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = layers.Dense(7)(x)
    
    out = KB.minimum(KB.maximum(x, gate_min), gate_max)
    
    rnn_model = Model(inputs=inputs, outputs=out)
    #rnn_model.summary()
    
    
    return rnn_model
