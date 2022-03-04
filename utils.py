import os
import inspect
import numpy as np

def load_parent_dir():
	currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	parentdir = os.path.dirname(currentdir)
	return parentdir

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric arguments, broadcasts it to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s

def identity(x): return x

def one_hot(y, n_labels, dtype=int):
        """Returns a matrix where each sample in y is represented as a row, 
            and each column represents the class label in the one-hot encoding scheme.
        Args:
            y (1d-array): the data to be encoded
            n_labels (int): the number of categories
        """
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    

def quantize(data, num_bins):
    bins = np.linspace(start=np.min(data), stop=np.max(data), num=num_bins, dtype=float)
    quantized = np.digitize(data, bins, right=True).astype(float)
    quantized *= (np.max(data) - np.min(data)) / (np.max(quantized) - np.min(quantized))   # scale the quantized data into the same size of the original data
    return quantized + np.min(data)  # add bias to the quantized data 


# %% Training Functions

def ridge(X, Y, ridge_noise):
    '''Compute readout weights matrix [Ridge Regression]
    Args:
        X : reservoir states matrix, (num_samples * N)
        Y : true outputs, (num_samples * output_dim)
        ridge_noise: the regularization parameter of ridge regression
    Returns:
        W_out: readout weights matrix, (output_dim * N)
    '''
    return np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T, X) + ridge_noise*np.eye(X.shape[1])))


def pinv(X, Y):
    '''Compute readout weights matrix [Moore-Penrose Pseudo Inverse]
    Args & Returns: Same as self._ridge()
    '''
    return np.dot(np.linalg.pinv(X), Y).T


