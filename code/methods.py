import numpy as np


def beta_ols(X, y):
	"""Calculate beta values using matrix inversion.
	
	Args:
        X (np.ndarray): design matrix
		y (np.ndarray): data to fit
		
	Returns:
        np.ndarray: beta values
	"""
	return np.linalg.pinv(X.T @ X) @ X.T @ y


def beta_ridge(X, y, lamb):
	"""Calculate beta ridge values using matrix inversion.
	
	Args:
        X (np.ndarray): design matrix
		y (np.ndarray): data to fit
		lamb (float): lambda value
		
	Returns:
        np.ndarray: beta values
	"""
	return (np.linalg.pinv(X.T @ X + lamb * np.identity(len(X[0]))) @ X.T @ y)


def k_fold(data, k):
    """Find indices for train and test data for k fold split.
    
    Args:
        data (np.ndarray): data set
		k (int): number of folds
		
	Returns:
        np.ndarray: tuples of indices for train and test data
    """
    n_samples = len(data)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    k_fold_indices = []

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        k_fold_indices.append((train_indices, test_indices))
        
    return k_fold_indices