import numpy as np
from pandas import DataFrame
from pandas import concat
import pandas as pd
import typing


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes



def graph_topology_5(input_sequence_length, forecast_horizon, sigma2, epsilon, train_len, val_len, test_len):
    #"""WS_S1	WS_S4	TWS_S25A	TWS_S25B	TWS_S26"""
    distance_adjcency = pd.read_csv('../data/distance_adjacency.csv', index_col=0)
    distance_adjcency.fillna(0, inplace=True)
    distance_adjcency.drop(['GATE_S25A', 'HWS_S25A', 'GATE_S25B', 'GATE_S25B2', 'HWS_S25B', 
                            'GATE_S26_1', 'GATE_S26_2', 'HWS_S26', 'MEAN_RAIN'], 
                             axis=1, inplace = True)
    distance_adjcency.drop(['GATE_S25A', 'HWS_S25A', 'GATE_S25B', 'GATE_S25B2', 'HWS_S25B', 
                            'GATE_S26_1', 'GATE_S26_2', 'HWS_S26', 'MEAN_RAIN'], 
                             axis=0, inplace = True)
    
    distance_adjcency_scaled = scale_distance(distance_adjcency)
    distance_adjcency_scaled.fillna(0, inplace=True)
    dis_adj = distance_adjcency_scaled.values
    
    adjacency_matrix = compute_adjacency_matrix(distance_adjcency, sigma2, epsilon)
    adjacency_matrix.iloc[0] = 0, 1, 1, 1, 1
    adjacency_matrix.iloc[1] = 1, 0, 1, 0, 0
    adjacency_matrix.iloc[2] = 1, 1, 0, 0, 0
    adjacency_matrix.iloc[3] = 1, 0, 0, 0, 1
    adjacency_matrix.iloc[4] = 1, 0, 0, 1, 0
    
    adjacency_matrix = np.array(adjacency_matrix)
    # adjacency_matrix.shape

    train_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], train_len, axis=0)
    val_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], val_len, axis=0)
    test_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], test_len, axis=0)

    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    print("node_indices:", node_indices, "\n" "neighbor_indices:", neighbor_indices)


    graph = GraphInfo(edges=(node_indices.tolist(), 
                      neighbor_indices.tolist()),
                      num_nodes=adjacency_matrix.shape[0])

    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
    
    
    return train_adjacency_matrix, val_adjacency_matrix, test_adjacency_matrix



def graph_topology_5_1(input_sequence_length, forecast_horizon, sigma2, epsilon, train_len, val_len, test_len):
    #"""WS_S1	WS_S4	TWS_S25A	TWS_S25B	TWS_S26"""
    distance_adjcency = pd.read_csv('data/distance_adjacency.csv', index_col=0)
    distance_adjcency.fillna(0, inplace=True)
    distance_adjcency.drop(['GATE_S25A', 'HWS_S25A', 'GATE_S25B', 'GATE_S25B2', 'HWS_S25B', 
                            'GATE_S26_1', 'GATE_S26_2', 'HWS_S26', 'MEAN_RAIN'], 
                             axis=1, inplace = True)
    distance_adjcency.drop(['GATE_S25A', 'HWS_S25A', 'GATE_S25B', 'GATE_S25B2', 'HWS_S25B', 
                            'GATE_S26_1', 'GATE_S26_2', 'HWS_S26', 'MEAN_RAIN'], 
                             axis=0, inplace = True)
    
    distance_adjcency_scaled = scale_distance(distance_adjcency)
    distance_adjcency_scaled.fillna(0, inplace=True)
    dis_adj = distance_adjcency_scaled.values
    
    adjacency_matrix = compute_adjacency_matrix(distance_adjcency, sigma2, epsilon)
    adjacency_matrix.iloc[0] = 0, 1, 1, 1, 1
    adjacency_matrix.iloc[1] = 1, 0, 1, 0, 0
    adjacency_matrix.iloc[2] = 1, 1, 0, 0, 0
    adjacency_matrix.iloc[3] = 1, 0, 0, 0, 1
    adjacency_matrix.iloc[4] = 1, 0, 0, 1, 0
    
    adjacency_matrix = np.array(adjacency_matrix)
    # adjacency_matrix.shape

    train_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], train_len, axis=0)
    val_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], val_len, axis=0)
    test_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], test_len, axis=0)

    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    print("node_indices:", node_indices, "\n" "neighbor_indices:", neighbor_indices)


    graph = GraphInfo(edges=(node_indices.tolist(), 
                      neighbor_indices.tolist()),
                      num_nodes=adjacency_matrix.shape[0])

    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
    
    
    return train_adjacency_matrix, val_adjacency_matrix, test_adjacency_matrix





def graph_topology(input_sequence_length, forecast_horizon, sigma2, epsilon, train_len, val_len, test_len):
    distance_adjcency = pd.read_csv('../data/distance_adjacency.csv', index_col=0)
    distance_adjcency.fillna(0, inplace=True)
    
    distance_adjcency_scaled = scale_distance(distance_adjcency)
    distance_adjcency_scaled.fillna(0, inplace=True)
    dis_adj = distance_adjcency_scaled.values
    
    adjacency_matrix = compute_adjacency_matrix(distance_adjcency, sigma2, epsilon)
    adjacency_matrix.iloc[0] = 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1
    adjacency_matrix.iloc[1] = 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[2] = 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[3] = 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[4] = 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[5] = 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[6] = 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[7] = 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1
    adjacency_matrix.iloc[8] = 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1
    adjacency_matrix.iloc[9] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
    adjacency_matrix.iloc[10] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1
    adjacency_matrix.iloc[11] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1
    adjacency_matrix.iloc[12] = 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1
    adjacency_matrix.iloc[13] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    
    adjacency_matrix = np.array(adjacency_matrix)
    # adjacency_matrix.shape

    train_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], train_len, axis=0)
    val_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], val_len, axis=0)
    test_adjacency_matrix = np.repeat(adjacency_matrix[np.newaxis, :, :], test_len, axis=0)

    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    print("node_indices:", node_indices, "\n" "neighbor_indices:", neighbor_indices)


    graph = GraphInfo(edges=(node_indices.tolist(), 
                      neighbor_indices.tolist()),
                      num_nodes=adjacency_matrix.shape[0])

    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
    
    
    return train_adjacency_matrix, val_adjacency_matrix, test_adjacency_matrix



def compute_adjacency_matrix(route_distances: np.ndarray, sigma2: float, epsilon: float):
    """Computes the adjacency matrix from distances matrix. 
    It uses formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to compute adjacency matrix.

    Args:
        route_distances: shape `(num_routes, num_routes)`. Entry `i,j` of this array is distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
                if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
                matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 100000.0
    w2 = route_distances * route_distances
    w_mask = np.ones([num_routes, num_routes]) - np.identity(num_routes)
    
    #return (np.exp(-w2 / sigma2) >= epsilon) * w_mask
    return np.exp(-w2 / sigma2) 


def scale_distance(distance_adjcency):
    max_adj = distance_adjcency.max()
    min_adj = distance_adjcency.min()
    distance_adjcency_scaled = (distance_adjcency - min_adj) / (max_adj - min_adj)
    return distance_adjcency_scaled

