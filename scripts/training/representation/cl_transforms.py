import numpy as np

def identity(array: np.array):
    return array

def reverse(array: np.array):
    """ Reverse the array with respect to time.

    Args:
        array (np.array): Input array of shape (batch_size x n_frames x n_channels x img_dim x img_dim)
    """
    return array[:, ::-1].copy() # copy, torch does not support negative stride

def add_gaussian_noise(array: np.array, mean: float = 0, std: float = 1):
    """ Add gaussian noise to every values of the array.

    Args:
        array (np.array): Input array of shape (batch_size x n_frames x n_channels x img_dim x img_dim)
        mean (float, optional): Mean of gaussian distribution
        std (float, optional): Standard deviation of gaussian distribution
    """
    return array + np.random.normal(mean, std, array.shape)

def permute_frames(array: np.array):
    """ Permute the frames independently for each video.

    Args:
        array (np.array): Input array of shape (batch_size x n_frames x n_channels x img_dim x img_dim)
    """
    batch_size, n_frames, _, _, _ = array.shape

    # one permutation per video
    ix = np.arange(batch_size * n_frames).reshape(batch_size, n_frames)
    np.apply_along_axis(np.random.shuffle, 1, ix)
    
    # reshape and permute frames
    per = array.reshape(batch_size * n_frames, -1)[ix.flatten()]
    
    return per.reshape(array.shape)

def jitter_channels(array: np.array, low: float = 0.5, high:float = 2):
    """ Randomly multiply the amplitude of the channels, independently for each video.

    Args:
        array (np.array): Input array of shape (batch_size x n_frames x n_channels x img_dim x img_dim)
        low (float, optional): Lowest multiplicator. Defaults to 0.5.
        high (float, optional): Highest multiplicator. Defaults to 2.
    """
    batch_size, _, n_channels, _, _ = array.shape
    
    # one random multiplicator per video, per channel
    jitter = np.random.uniform(size = batch_size * n_channels).reshape(n_channels, batch_size)
    
    # make the channels the second axis, transpose, broadcast and revert to original axis
    return (array.swapaxes(1, 2).T * jitter).T.swapaxes(1, 2)