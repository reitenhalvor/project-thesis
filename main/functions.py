import numpy as np

"""
Script that defines helper functions that should be globally available to all notebooks. 
"""

def RMSE(targets, predictions):
    """
    Calculates the mean absolute error between vectors/matrices
    
    :param targets: a matrix/vector of true targets
    :param predictions: a matrix/vector of predictions
    """
    return np.sqrt(np.mean(np.square(targets - predictions)))


def MAE(targets, predictions, vector=False):
    """
    Calculates the mean absolute error between vectors/matrices
    
    :param targets: a matrix/vector of true targets
    :param predictions: a matrix/vector of predictions
    :param vector: boolean stating if a vector of MAEs should be returned in the case where the targets/predictions are matrices.
    """
    if vector:
        return np.mean(np.abs(targets - predictions), axis=0)
    else:
        return np.mean(np.abs(targets - predictions))
    
    
def batch_generator(x_data, y_data, lookback, batch_size=128):
    """
    Generator function for creating random batches of training-data.
    
    :param x_data: Numpy array of the features, not targets. 2D array, normalized/scaled, numpy.
    :param y_data: Numpy array of the targets, not features. 2D array, normalized/scaled, numpy. 
    :param lookback: How many timesteps back the input data should go.
    :param batch_size: The number of samples per batch.
    """
    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, lookback, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, lookback, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(x_data.shape[0] - lookback)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_data[idx:idx+lookback]
            y_batch[i] = y_data[idx:idx+lookback]
        
        yield (x_batch, y_batch)

        