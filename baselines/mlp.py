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
from tensorflow.keras.layers import SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as KB



def mlp(input_shape, mlp_unit1, mlp_unit2, l1_reg, l2_reg, dropout, masked_value):
    
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Dense(mlp_unit1, 
              activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
             )(masked_inputs)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit2, 
              activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
             )(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(96)(x)

    mlp_model = Model(inputs=inputs, outputs=outputs)
    #mlp_model.summary()


    return mlp_model



def mlp_gate_generator(input_shape, mlp_unit1, mlp_unit2, l1_reg, l2_reg, dropout, masked_value):

    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Dense(mlp_unit1, 
              activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
             )(masked_inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit2, 
              activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
             )(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = layers.Dense(7)(x)
    
    mlp_model = Model(inputs=inputs, outputs=x)
    #mlp_model.summary()


    return mlp_model




def mlp_gate_generator_1(input_shape, mlp_unit1, mlp_unit2, mlp_unit3, l1_reg, l2_reg, dropout, masked_value, gate_min, gate_max):

    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Dense(mlp_unit1, 
#               activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
             )(masked_inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit2, 
#               activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
             )(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dense(mlp_unit3, 
#               activation='relu', 
              kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
             )(x)
    x = layers.Dense(7)(x)
    out = KB.minimum(KB.maximum(x, gate_min), gate_max)
    
    mlp_model = Model(inputs=inputs, outputs=out)
    #mlp_model.summary()


    return mlp_model


