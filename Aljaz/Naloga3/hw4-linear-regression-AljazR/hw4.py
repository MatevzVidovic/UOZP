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

    return (np.sum((X @ theta - y)**2) + l2_lambda * np.sum(theta[1:]**2)) / (2 * len(y))


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

    return  ((X.T @ (X @ theta - y)) + l2_lambda * np.hstack((0, theta[1:]))) / len(y)


def gradient_descent(X, y, l2_lambda, lr=0.01, tol=1e-13, max_iter=100_000):
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

    theta = np.zeros(X.shape[1])
    J = cost(X, y, theta, l2_lambda)
    for _ in range(max_iter):
        theta -= lr * gradient(X, y, theta, l2_lambda)
        
        J_new = cost(X, y, theta, l2_lambda)
        if np.abs(J - J_new) < tol:
            break
        J = J_new

    return theta


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
        X_ones = np.hstack((np.ones((X.shape[0], 1)), X))

        theta = gradient_descent(X_ones, y, self.l2_lambda)

        self.intercept = theta[0]
        self.coefs = theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Parameters
        ----------
        X: Test data of shape (n_samples, n_features)
            
        Returns
        -------
            Returns predicted values of shape (n_samples,)
        """
        return X @ self.coefs + self.intercept
