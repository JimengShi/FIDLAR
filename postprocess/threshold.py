from pandas import DataFrame
from pandas import concat
import numpy as np



def flood_threshold(water, upper_threshold):
    """
    flood_threshold.
    Arguments:
        water: 3D array (# sample, K, # stations)
    Returns:
        flooding time steps and flooding areas.
    """
    time_steps = water > upper_threshold
    count_time_steps = np.count_nonzero(time_steps)
    
    flood_area = 0
    for z in range(len(water)):
        one_sample_diff = water[z, :, :] - upper_threshold

        for i in range(one_sample_diff.shape[0]):
            for j in range(one_sample_diff.shape[1]):
                if one_sample_diff[i][j] > 0:
                    flood_area += one_sample_diff[i][j]
    
    print("time steps: {}, areas: {}".format(count_time_steps, flood_area))
    
    return



def drought_threshold(water, lower_threshold):
    """
    flood_threshold.
    Arguments:
        water: 3D array (# sample, K, # stations)
    Returns:
        flooding time steps and flooding areas.
    """
    time_steps = water < lower_threshold
    count_time_steps = np.count_nonzero(time_steps)
    
    flood_area = 0
    for z in range(len(water)):
        one_sample_diff = water[z, :, :] - lower_threshold

        for i in range(one_sample_diff.shape[0]):
            for j in range(one_sample_diff.shape[1]):
                if one_sample_diff[i][j] < 0:
                    flood_area += abs(one_sample_diff[i][j])
    
    print("time steps: {}, areas: {}".format(count_time_steps, flood_area))
    
    return



def flood_threshold_t1(water, time_step, upper_threshold):
    """
    flood_threshold.
    Arguments:
        water: 3D array (# sample, K, # stations)
    Returns:
        flooding time steps and flooding areas.
    """
    time_steps1 = water[:, time_step, 0] > upper_threshold
    time_steps25a = water[:, time_step, 1] > upper_threshold
    time_steps25b = water[:, time_step, 2] > upper_threshold
    time_steps26 = water[:, time_step, 3] > upper_threshold
    count_time_steps = np.count_nonzero(time_steps1) + np.count_nonzero(time_steps25a) + np.count_nonzero(time_steps25b) + np.count_nonzero(time_steps26)
    
    flood_area = 0
    water_1 = water[:, time_step, 0]
    water_25a = water[:, time_step, 1]
    water_25b = water[:, time_step, 2]
    water_26 = water[:, time_step, 3]
    
    flood_area_1 = 0
    for i in range(len(water)):
        one_sample_diff_1 = water_1[i] - upper_threshold
        if one_sample_diff_1 > 0:
            flood_area_1 += one_sample_diff_1
            
    flood_area_25a = 0
    for i in range(len(water)):
        one_sample_diff_25a = water_25a[i] - upper_threshold
        if one_sample_diff_25a > 0:
            flood_area_25a += one_sample_diff_25a
            
    flood_area_25b = 0
    for i in range(len(water)):
        one_sample_diff_25b = water_25b[i] - upper_threshold
        if one_sample_diff_25b > 0:
            flood_area_25b += one_sample_diff_25b
            
    flood_area_26 = 0
    for i in range(len(water)):
        one_sample_diff_26 = water_26[i] - upper_threshold
        if one_sample_diff_26 > 0:
            flood_area_26 += one_sample_diff_26
    
    flood_area = flood_area_1 + flood_area_25a + flood_area_25b + flood_area_26
    
    print("S1, S25A, S25B, S26 time steps: {}, {}, {}, {}".format(np.count_nonzero(time_steps1), np.count_nonzero(time_steps25a), np.count_nonzero(time_steps25b), np.count_nonzero(time_steps26)))
    print("S1, S25A, S25B, S26 areas: {}, {}, {}, {}".format(round(flood_area_1, 4), round(flood_area_25a, 4), round(flood_area_25b), round(flood_area_26, 4)))
    print("TOTAL time steps: {}; TOTAL areas: {}".format(count_time_steps, round(flood_area, 4)))
    print("--------------------------------------------------")
          
    return



def drought_threshold_t1(water, time_step, lower_threshold):
    """
    flood_threshold.
    Arguments:
        water: 3D array (# sample, K, # stations)
    Returns:
        flooding time steps and flooding areas.
    """
    time_steps1 = water[:, time_step, 0] < lower_threshold  
    time_steps25a = water[:, time_step, 1] < lower_threshold
    time_steps25b = water[:, time_step, 2] < lower_threshold
    time_steps26 = water[:, time_step, 3] < lower_threshold
    count_time_steps = np.count_nonzero(time_steps1) + np.count_nonzero(time_steps25a) + np.count_nonzero(time_steps25b) + np.count_nonzero(time_steps26)
    
    
    drought_area = 0
    water_1 = water[:, time_step, 0]
    water_25a = water[:, time_step, 1]
    water_25b = water[:, time_step, 2]
    water_26 = water[:, time_step, 3]
    
    drought_area_1 = 0
    for i in range(len(water)):
        one_sample_diff_1 = water_1[i] - lower_threshold
        if one_sample_diff_1 < 0:
            drought_area_1 += one_sample_diff_1
            
    drought_area_25a = 0
    for i in range(len(water)):
        one_sample_diff_25a = water_25a[i] - lower_threshold
        if one_sample_diff_25a < 0:
            drought_area_25a += one_sample_diff_25a
            
    drought_area_25b = 0
    for i in range(len(water)):
        one_sample_diff_25b = water_25b[i] - lower_threshold
        if one_sample_diff_25b < 0:
            drought_area_25b += one_sample_diff_25b
            
    drought_area_26 = 0
    for i in range(len(water)):
        one_sample_diff_26 = water_26[i] - lower_threshold
        if one_sample_diff_26 < 0:
            drought_area_26 += one_sample_diff_26
    
    drought_area = drought_area_1 + drought_area_25a + drought_area_25b + drought_area_26
    
    print("S1, S25A, S25B, S26 time steps: {}, {}, {}, {}".format(np.count_nonzero(time_steps1), np.count_nonzero(time_steps25a), np.count_nonzero(time_steps25b), np.count_nonzero(time_steps26)))
    print("S1, S25A, S25B, S26 areas: {}, {}, {}, {}:".format(round(drought_area_1, 4), round(drought_area_25a, 4), round(drought_area_25b, 4), round(drought_area_26, 4)))
    print("TOTAL time steps: {}; TOTAL areas: {}".format(count_time_steps, round(drought_area, 4)))
    print("--------------------------------------------------")
    
    return
