import numpy as np
import pandas as pd
from pandas import DataFrame, concat, read_csv
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from preprocess.helper import series_to_supervised, stage_series_to_supervised


def graph_water_transformer_cov_process(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('../data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_hws_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    # ==================== water & covariates ====================
    train_tws = np.concatenate((train_X_mask[:, :n_hours, 1:2], train_X_mask[:, :n_hours, -4:]), axis=-1)
    val_tws = np.concatenate((val_X_mask[:, :n_hours, 1:2], val_X_mask[:, :n_hours, -4:]), axis=-1)
    test_tws = np.concatenate((test_X_mask[:, :n_hours, 1:2], test_X_mask[:, :n_hours, -4:]), axis=-1)


    train_cov = train_X_mask[:, :, :12]
    val_cov = val_X_mask[:, :, :12]
    test_cov = test_X_mask[:, :, :12]
    
    
    train_tws_reshape = train_tws.reshape((-1, 5, 72))
    val_tws_reshape = val_tws.reshape((-1, 5, 72))
    test_tws_reshape = test_tws.reshape((-1, 5, 72))
    
    return train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler




def graph_water_transformer_cov_process_1(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_hws_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    # ==================== water & covariates ====================
    train_tws = np.concatenate((train_X_mask[:, :n_hours, 1:2], train_X_mask[:, :n_hours, -4:]), axis=-1)
    val_tws = np.concatenate((val_X_mask[:, :n_hours, 1:2], val_X_mask[:, :n_hours, -4:]), axis=-1)
    test_tws = np.concatenate((test_X_mask[:, :n_hours, 1:2], test_X_mask[:, :n_hours, -4:]), axis=-1)


    train_cov = train_X_mask[:, :, :12]
    val_cov = val_X_mask[:, :, :12]
    test_cov = test_X_mask[:, :, :12]
    
    
    train_tws_reshape = train_tws.reshape((-1, 5, 72))
    val_tws_reshape = val_tws.reshape((-1, 5, 72))
    test_tws_reshape = test_tws.reshape((-1, 5, 72))
    
    return train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler





def graph_water_transformer_cov_process_for_gate_predictor(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('../data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_gate_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    gate_true = past_future[:, n_hours:, mask_gate_start_index:mask_pump_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    gate_true_reshape = gate_true.reshape((gate_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
    train_gate_true = gate_true_reshape[:split1]
    val_gate_true = gate_true_reshape[split1:split2]
    test_gate_true = gate_true_reshape[split1:]
    

    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    gate_scaler = MinMaxScaler(feature_range=(0, 1))
    train_gate_true_scaled = gate_scaler.fit_transform(train_gate_true)
    val_gate_true_scaled = gate_scaler.fit_transform(val_gate_true)
    test_gate_true_scaled = gate_scaler.fit_transform(test_gate_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
    train_gate_pump_y = train_gate_true_scaled
    val_gate_pump_y = val_gate_true_scaled
    test_gate_pump_y = test_gate_true_scaled
    
    train_gate_pump_y = train_gate_pump_y.reshape((-1, 24, 7))
    val_gate_pump_y = val_gate_pump_y.reshape((-1, 24, 7))
    test_gate_pump_y = test_gate_pump_y.reshape((-1, 24, 7))

    
    # ==================== water & covariates ====================
    train_tws = np.concatenate((train_X_mask[:, :n_hours, 1:2], train_X_mask[:, :n_hours, -4:]), axis=-1)
    val_tws = np.concatenate((val_X_mask[:, :n_hours, 1:2], val_X_mask[:, :n_hours, -4:]), axis=-1)
    test_tws = np.concatenate((test_X_mask[:, :n_hours, 1:2], test_X_mask[:, :n_hours, -4:]), axis=-1)
    
    
    train_tws_reshape = train_tws.reshape((-1, 5, 72))
    val_tws_reshape = val_tws.reshape((-1, 5, 72))
    test_tws_reshape = test_tws.reshape((-1, 5, 72))
    
    
    train_cov = train_X_mask[:, :, :12]
    val_cov = val_X_mask[:, :, :12]
    test_cov = test_X_mask[:, :, :12]
    
    print('train_tws/val_tws/test_tws:', train_tws_reshape.shape, val_tws_reshape.shape, test_tws_reshape.shape, '\n',
          'train_cov/val_cov/test_cov:', train_cov.shape, val_cov.shape, test_cov.shape, '\n',
          'train_ws_y/val_ws_y/test_ws_y:', train_ws_y.shape, val_ws_y.shape, test_ws_y.shape, '\n',
          ' train_gate_pump_y/val_gate_pump_y/test_gate_pump_y:', train_gate_pump_y.shape, val_gate_pump_y.shape, test_gate_pump_y.shape
          
         )
    
    return train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_gate_pump_y, val_gate_pump_y, test_gate_pump_y, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler, gate_scaler



def graph_water_transformer_cov_process_for_gate_predictor_1(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_gate_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    gate_true = past_future[:, n_hours:, mask_gate_start_index:mask_pump_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    gate_true_reshape = gate_true.reshape((gate_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
    train_gate_true = gate_true_reshape[:split1]
    val_gate_true = gate_true_reshape[split1:split2]
    test_gate_true = gate_true_reshape[split1:]
    

    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    gate_scaler = MinMaxScaler(feature_range=(0, 1))
    train_gate_true_scaled = gate_scaler.fit_transform(train_gate_true)
    val_gate_true_scaled = gate_scaler.fit_transform(val_gate_true)
    test_gate_true_scaled = gate_scaler.fit_transform(test_gate_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
    train_gate_pump_y = train_gate_true_scaled
    val_gate_pump_y = val_gate_true_scaled
    test_gate_pump_y = test_gate_true_scaled
    
    train_gate_pump_y = train_gate_pump_y.reshape((-1, 24, 7))
    val_gate_pump_y = val_gate_pump_y.reshape((-1, 24, 7))
    test_gate_pump_y = test_gate_pump_y.reshape((-1, 24, 7))

    
    # ==================== water & covariates ====================
    train_tws = np.concatenate((train_X_mask[:, :n_hours, 1:2], train_X_mask[:, :n_hours, -4:]), axis=-1)
    val_tws = np.concatenate((val_X_mask[:, :n_hours, 1:2], val_X_mask[:, :n_hours, -4:]), axis=-1)
    test_tws = np.concatenate((test_X_mask[:, :n_hours, 1:2], test_X_mask[:, :n_hours, -4:]), axis=-1)
    
    
    train_tws_reshape = train_tws.reshape((-1, 5, 72))
    val_tws_reshape = val_tws.reshape((-1, 5, 72))
    test_tws_reshape = test_tws.reshape((-1, 5, 72))
    
    
    train_cov = train_X_mask[:, :, :12]
    val_cov = val_X_mask[:, :, :12]
    test_cov = test_X_mask[:, :, :12]
    
    print('train_tws/val_tws/test_tws:', train_tws_reshape.shape, val_tws_reshape.shape, test_tws_reshape.shape, '\n',
          'train_cov/val_cov/test_cov:', train_cov.shape, val_cov.shape, test_cov.shape, '\n',
          'train_ws_y/val_ws_y/test_ws_y:', train_ws_y.shape, val_ws_y.shape, test_ws_y.shape, '\n',
          ' train_gate_pump_y/val_gate_pump_y/test_gate_pump_y:', train_gate_pump_y.shape, val_gate_pump_y.shape, test_gate_pump_y.shape
          
         )
    
    return train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_gate_pump_y, val_gate_pump_y, test_gate_pump_y, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler, gate_scaler



def graph_global_transformer_local_process(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('../data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_hws_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    # ==================== each station ====================
    # S26
    rain_26_idx, gate_26_idx, pump_26_idx, hws_26_idx, tws_26_idx = 0, 5, 8, 11, 15
    train_26 = np.concatenate( (train_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                                train_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                                train_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                train_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                                train_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                               ), axis=-1 
                             )
    val_26 = np.concatenate( (val_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                              val_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                              val_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                              val_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                              val_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                             ), axis=-1 
                           )
    test_26 = np.concatenate( (test_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                               test_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                               test_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                               test_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                               test_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                              ), axis=-1 
                            )


    # S25B
    rain_25b_idx, gate_25b_idx, pump_25b_idx, hws_25b_idx, tws_25b_idx = 0, 3, 7, 10, 14
    train_25B = np.concatenate( (train_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                                 train_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                                 train_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                 train_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                                 train_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                                ), axis=-1 
                              )
    val_25B = np.concatenate( (val_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                               val_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                               val_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                               val_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                               val_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                              ), axis=-1 
                            )
    test_25B = np.concatenate( (test_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                                test_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                                test_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                test_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                                test_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                               ), axis=-1 
                             )


    # S25A
    rain_25a_idx, gate_25a_idx, hws_25a_idx, tws_25a_idx = 0, 2, 9, 13
    train_25A = np.concatenate( (train_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                                 train_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                                 train_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                                 train_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                                ), axis=-1 
                              )
    val_25A = np.concatenate( (val_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                               val_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                               val_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                               val_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                              ), axis=-1 
                            )
    test_25A = np.concatenate( ( test_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                                 test_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                                 test_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                                 test_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                               ), axis=-1 
                             )


    # S1
    rain_1_idx, tws_1_idx = 0, 12
    train_1 = np.concatenate((train_X_mask[:, :, rain_1_idx:rain_1_idx+1], train_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)
    val_1 = np.concatenate((val_X_mask[:, :, rain_1_idx:rain_1_idx+1], val_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)
    test_1 = np.concatenate((test_X_mask[:, :, rain_1_idx:rain_1_idx+1], test_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)


    # S4
    rain_4_idx, tws_4_idx = 0, 1
    train_4 = np.concatenate((train_X_mask[:, :, rain_4_idx:rain_4_idx+1], train_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    val_4 = np.concatenate((val_X_mask[:, :, rain_4_idx:rain_4_idx+1], val_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    test_4 = np.concatenate((test_X_mask[:, :, rain_4_idx:rain_4_idx+1], test_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    
    
    return train_26, val_26, test_26, train_25B, val_25B, test_25B, train_25A, val_25A, test_25A, train_1, val_1, test_1, train_4, val_4, test_4, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler



def graph_global_transformer_local_process_for_gate_predictor(n_hours, K, masked_value, split_1, split_2):
    
    # ==================== import dataset ====================
    dataset = pd.read_csv('../data/Merged-update_hourly.csv', index_col=0)
    dataset.fillna(0, inplace=True)
    print(dataset.columns)
    
    
    # ==================== convert dataset to supervised mode ====================
    data = dataset[['MEAN_RAIN', 'WS_S4',
                    'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                    'PUMP_S25B', 'PUMP_S26',
                    #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                    'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                    'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
    features = data.shape[1]
    
    #print("data.shape:", data.shape)

    data_supervised = series_to_supervised(data, n_hours, K)
    #print("data_supervised.shape:", data_supervised.shape)
    
    
    col_names = ['MEAN_RAIN', 'WS_S4',
                 'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                 'PUMP_S25B', 'PUMP_S26',
                  #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26',
                 'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                 'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26'] * (n_hours+K)
    
    data_supervised.reset_index(drop=True, inplace=True)
    data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
    #print("data_supervised:", data_supervised)
    
    
    # ==================== past & future ====================
    past = data_supervised.iloc[:, :n_hours*data.shape[1]]
    past = past.to_numpy(dtype='float32')
    past = past.reshape((-1, n_hours, data.shape[1]))
    
    future = data_supervised.iloc[:, n_hours*data.shape[1]:]
    future = future.to_numpy(dtype='float32')
    future = future.reshape((-1, K, data.shape[1]))
    
    past_future = np.concatenate((past, future), axis=1)
    past_future = past_future.astype(np.float32)
    
    
    # ==================== masking ====================
    mask_gate_start_index = 2
    mask_gate_end_index = 6
    mask_pump_start_index = 7
    mask_pump_end_index = 8
    mask_hws_start_index = 9
    mask_hws_end_index = 11
    mask_tws_start_index = 12
    mask_tws_end_index = 15
    
    past_future_mask = past_future.copy()
    past_future_mask[:, n_hours:, mask_gate_start_index:mask_tws_end_index+1] = masked_value  # masking ws
    
    X_mask = past_future_mask
    ws_true = past_future[:, n_hours:, mask_tws_start_index:mask_tws_end_index+1]
    gate_true = past_future[:, n_hours:, mask_gate_start_index:mask_pump_end_index+1]
    
    X_mask_reshape = X_mask.reshape((X_mask.shape[0], -1))
    ws_true_reshape = ws_true.reshape((ws_true.shape[0], -1))
    gate_true_reshape = gate_true.reshape((gate_true.shape[0], -1))
    
    split1 = int(len(X_mask_reshape)*split_1)
    split2 = int(len(X_mask_reshape)*split_2)
    
    
    # train / val / test
    train_X_mask = X_mask_reshape[:split1]
    val_X_mask = X_mask_reshape[split1:split2]
    test_X_mask = X_mask_reshape[split1:]

    train_ws_true = ws_true_reshape[:split1]
    val_ws_true = ws_true_reshape[split1:split2]
    test_ws_true = ws_true_reshape[split1:]
    
    train_gate_true = gate_true_reshape[:split1]
    val_gate_true = gate_true_reshape[split1:split2]
    test_gate_true = gate_true_reshape[split1:]
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    
    # ==================== normalization ====================
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X_mask_scaled = scaler.fit_transform(train_X_mask)
    val_X_mask_scaled = scaler.fit_transform(val_X_mask)
    test_X_mask_scaled = scaler.fit_transform(test_X_mask)


    ws_scaler = MinMaxScaler(feature_range=(0, 1))
    train_ws_true_scaled = ws_scaler.fit_transform(train_ws_true)
    val_ws_true_scaled = ws_scaler.fit_transform(val_ws_true)
    test_ws_true_scaled = ws_scaler.fit_transform(test_ws_true)
    
    
    gate_scaler = MinMaxScaler(feature_range=(0, 1))
    train_gate_true_scaled = gate_scaler.fit_transform(train_gate_true)
    val_gate_true_scaled = gate_scaler.fit_transform(val_gate_true)
    test_gate_true_scaled = gate_scaler.fit_transform(test_gate_true)
    
    
    # final train / val / test
    train_X_mask = train_X_mask_scaled.reshape((-1, n_hours+K, features))
    val_X_mask = val_X_mask_scaled.reshape((-1, n_hours+K, features))
    test_X_mask = test_X_mask_scaled.reshape((-1, n_hours+K, features))

    train_ws_y = train_ws_true_scaled
    val_ws_y = val_ws_true_scaled
    test_ws_y = test_ws_true_scaled
    
    train_gate_y = train_gate_true_scaled
    val_gate_y = val_gate_true_scaled
    test_gate_y = test_gate_true_scaled
    
#     print('Train/val/test X with mask:', train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, '\n',
#           'Train/val/test ws true:', train_ws_true.shape, val_ws_true.shape, test_ws_true.shape
#          )
    
    # ==================== each station ====================
    # S26
    rain_26_idx, gate_26_idx, pump_26_idx, hws_26_idx, tws_26_idx = 0, 5, 8, 11, 15
    train_26 = np.concatenate( (train_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                                train_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                                train_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                train_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                                train_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                               ), axis=-1 
                             )
    val_26 = np.concatenate( (val_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                              val_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                              val_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                              val_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                              val_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                             ), axis=-1 
                           )
    test_26 = np.concatenate( (test_X_mask[:, :, rain_26_idx:rain_26_idx+1], 
                               test_X_mask[:, :, gate_26_idx:gate_26_idx+2],
                               test_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                               test_X_mask[:, :, hws_26_idx:hws_26_idx+1],
                               test_X_mask[:, :, tws_26_idx:tws_26_idx+1],
                              ), axis=-1 
                            )


    # S25B
    rain_25b_idx, gate_25b_idx, pump_25b_idx, hws_25b_idx, tws_25b_idx = 0, 3, 7, 10, 14
    train_25B = np.concatenate( (train_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                                 train_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                                 train_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                 train_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                                 train_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                                ), axis=-1 
                              )
    val_25B = np.concatenate( (val_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                               val_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                               val_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                               val_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                               val_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                              ), axis=-1 
                            )
    test_25B = np.concatenate( (test_X_mask[:, :, rain_25b_idx:rain_25b_idx+1], 
                                test_X_mask[:, :, gate_25b_idx:gate_25b_idx+2], 
                                test_X_mask[:, :, pump_26_idx:pump_26_idx+1],
                                test_X_mask[:, :, hws_25b_idx:hws_25b_idx+1],
                                test_X_mask[:, :, tws_25b_idx:tws_25b_idx+1],
                               ), axis=-1 
                             )


    # S25A
    rain_25a_idx, gate_25a_idx, hws_25a_idx, tws_25a_idx = 0, 2, 9, 13
    train_25A = np.concatenate( (train_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                                 train_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                                 train_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                                 train_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                                ), axis=-1 
                              )
    val_25A = np.concatenate( (val_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                               val_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                               val_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                               val_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                              ), axis=-1 
                            )
    test_25A = np.concatenate( ( test_X_mask[:, :, rain_25a_idx:rain_25a_idx+1], 
                                 test_X_mask[:, :, gate_25a_idx:gate_25a_idx+1], 
                                 test_X_mask[:, :, hws_25a_idx:hws_25a_idx+1],
                                 test_X_mask[:, :, tws_25a_idx:tws_25a_idx+1],
                               ), axis=-1 
                             )


    # S1
    rain_1_idx, tws_1_idx = 0, 12
    train_1 = np.concatenate((train_X_mask[:, :, rain_1_idx:rain_1_idx+1], train_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)
    val_1 = np.concatenate((val_X_mask[:, :, rain_1_idx:rain_1_idx+1], val_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)
    test_1 = np.concatenate((test_X_mask[:, :, rain_1_idx:rain_1_idx+1], test_X_mask[:, :, tws_1_idx:tws_1_idx+1]), axis=-1)


    # S4
    rain_4_idx, tws_4_idx = 0, 1
    train_4 = np.concatenate((train_X_mask[:, :, rain_4_idx:rain_4_idx+1], train_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    val_4 = np.concatenate((val_X_mask[:, :, rain_4_idx:rain_4_idx+1], val_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    test_4 = np.concatenate((test_X_mask[:, :, rain_4_idx:rain_4_idx+1], test_X_mask[:, :, tws_4_idx:tws_4_idx+1]), axis=-1)
    
    
    return train_26, val_26, test_26, train_25B, val_25B, test_25B, train_25A, val_25A, test_25A, train_1, val_1, test_1, train_4, val_4, test_4, train_ws_y, val_ws_y, test_ws_y, train_gate_y, val_gate_y, test_gate_y, scaler, ws_scaler, gate_scaler