from pandas import DataFrame
from pandas import concat
import numpy as np



def estimate_error(error):
    """
    flood_threshold.
    Arguments:
        water: 3D array (# sample, K, # stations)
    Returns:
        # of over estimate, # of under estimate
    """
    over_estimate = error > 0
    num_over_estimate = np.count_nonzero(over_estimate)
    
    under_estimate = error < 0
    num_under_estimate = np.count_nonzero(under_estimate)
    
    
    
    return num_over_estimate, num_under_estimate
