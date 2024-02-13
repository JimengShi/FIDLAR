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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as KB



def cnn(input_shape, cnn_unit1, cnn_unit2, cnn_unit3, kernel_size, dropout, pool_size, masked_value):
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Conv1D(cnn_unit1, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(masked_inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit2, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Conv1D(cnn_unit3, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Flatten()(x)

    outputs = Dense(96)(x)

    cnn_model = Model(inputs=inputs, outputs=outputs)
    #cnn_model.summary()


    return cnn_model


def cnn_gate_generator(input_shape, cnn_unit1, cnn_unit2, cnn_unit3, cnn_unit4, kernel_size, dropout, pool_size, masked_value):
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Conv1D(cnn_unit1, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(masked_inputs)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit2, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit3, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit4, 
               kernel_size=kernel_size,
               padding="same",
              )(x)


    cnn_model = Model(inputs=inputs, outputs=x)
    #cnn_model.summary()


    return cnn_model


def cnn_gate_generator_1(input_shape, cnn_unit1, cnn_unit2, cnn_unit3, cnn_unit4, kernel_size, dropout, pool_size, masked_value, gate_min, gate_max):
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Conv1D(cnn_unit1, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(masked_inputs)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit2, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit3, 
               kernel_size=kernel_size, 
#                activation='relu', 
               padding="same",
              )(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit4, 
               kernel_size=kernel_size,
               padding="same",
               #activation='relu'
              )(x)

    out = KB.minimum(KB.maximum(x, gate_min), gate_max)


    cnn_model = Model(inputs=inputs, outputs=out)
    #cnn_model.summary()


    return cnn_model
