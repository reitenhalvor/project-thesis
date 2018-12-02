import numpy as np

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
    """
    A python generator to generate batches to train on.
    Intuitive description: Given data going as far back as `lookback` timesteps (a timestep is for instance 10 minutes)
    and sampled every `step` timesteps, can you predict the output in `delay` timesteps?

    :param data: Numpy array of all the data, both predictors and targets. 2D array, normalized/scaled, numpy.
    :param lookback: How many timesteps back the input data should go.
    :param delay: How many timesteps in the future the target should be.
    :param min_index: Indices in the data array that delimit which timesteps to draw from.
            This is useful for keeping a segment of the data for validation and another for testing.
    :param max_index: See min_index.
    :param shuffle: Whether to shuffle the samples or draw them in chronological order.
    :param batch_size: The number of samples per batch.
    :param step: The period, in timesteps, at which you sample data.

    :return: python generator, the next batches of samples and targets
    """

    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))

        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay,-1]

        yield samples, targets
