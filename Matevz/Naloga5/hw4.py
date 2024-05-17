import numpy as np


def cost(X, y, theta, l2_lambda):
    """ A cost function.

    Parameters
    ----------
    X: Training data of shape (n_samples, n_features)
    y: Target values of shape (n_samples,)
    theta: Parameters of shape (n_features,)
    l2_lambda: L2 regularization parameter
        
    Returns
    -------
        The value of the cost function
    """
    raise NotImplementedError


def gradient(X, y, theta, l2_lambda):
    """ Gradient of cost function.

    Parameters
    ----------
    X: Training data (n_samples, n_features)
    y: Target valuese (n_samples,)
    theta: Parameters of shape (n_features,)
    l2_lambda: L2 regularization parameter
        
    Returns
    -------
        Gradient of shape (n_features,)
    """
    raise NotImplementedError


def gradient_descent(
        X, 
        y,
        l2_lambda,
        lr=0.01,
        tol=1e-6, 
        max_iter=100_000):
    """ Implementation of gradient descent.

    Parameters
    ----------
    X: Training data of shape (n_samples, n_features)
    y: Target values of shape (n_samples,)
    l2_lambda: L2 regularization parameter
    lr: The learning rate.
    tol: The stopping criterion (tolerance).
    max_iter: The maximum number of passes (aka epochs).

    Returns
    -------
        The parameters theta of shape (n_features,)
    """
    raise NotImplementedError


class LinearRegression:
    def __init__(self, l2_lambda = 0):
        self.coefs = None
        self.intercept = None
        self.l2_lambda = l2_lambda
        

    def fit(self, X, y):
        """
        The fit method of LinearRegression accepts X and y
        as input and save the coefficients of the linear model.

        Parameters
        ----------
        X: Training data of shape (n_samples, n_features)
        y: Target values of shape (n_samples,)
            
        Returns
        -------
            None
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Parameters
        ----------
        X: Test data of shape (n_samples, n_features)
            
        Returns
        -------
            Returns predicted values of shape (n_samples,)
        """
        raise NotImplementedError
