import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def integrated_gradients(model, input_data, baseline_data, num_steps=50):
    # Convert input and baseline data to tensors
    input_data = tf.convert_to_tensor(input_data)
    baseline_data = tf.convert_to_tensor(baseline_data)

    # Compute the gradients of the model's output with respect to the input
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        predictions = model(input_data)
    gradients = tape.gradient(predictions, input_data)

    # Calculate the difference between input and baseline
    input_diff = input_data - baseline_data

    # Initialize an array to store the integrated gradients
    integrated_gradients = np.zeros_like(input_data)

    # Compute integrated gradients for each step
    for step in range(num_steps):
        # Interpolate between baseline and input
        interpolated_input = baseline_data + (input_diff * step / num_steps)

        # Compute gradients at the interpolated input
        with tf.GradientTape() as tape:
            tape.watch(interpolated_input)
            interpolated_predictions = model(interpolated_input)
        interpolated_gradients = tape.gradient(interpolated_predictions, interpolated_input)

        # Accumulate the gradients to compute integrated gradients
        integrated_gradients += interpolated_gradients

    # Average the accumulated gradients and multiply by input difference
    integrated_gradients *= input_diff / num_steps

    # Return the computed integrated gradients
    return integrated_gradients



def colorline(x, y, z=None, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    segments = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([segments[:-1], segments[1:]], axis=1)

    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    plt.gca().add_collection(lc)

    return lc