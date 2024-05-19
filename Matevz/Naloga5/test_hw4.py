import unittest
import ast
import inspect
import time
import numpy as np


def uses_loop(function):
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if isinstance(node, ast.Name) and node.id == "map":
            return True
        elif isinstance(node, (ast.For, ast.While, ast.ListComp)):
            return True
        # Throw error also if NotImplementedError is raised
        elif isinstance(node, ast.Raise):
            return True


class A_TestOutputValues(unittest.TestCase):

    def setUp(self) -> None:
        rng = np.random.RandomState(0)
        self.y = rng.randn(1)
        self.X = rng.randn(1, 1)

    def test_010_fit_return(self):
        from hw4 import LinearRegression

        lr = LinearRegression()
        fit_return = lr.fit(self.X, self.y)

        self.assertIsNone(fit_return, f"fit() method should return None, but got: {type(fit_return)}")
        self.assertIsNotNone(lr.coefs, "fit() method should set self.coefs")
        self.assertIsNotNone(lr.intercept, "fit() method should set self.intercept")

    def test_020_predict_return(self):
        from hw4 import LinearRegression

        lr = LinearRegression()
        lr.fit(self.X, self.y)
        predict_return = lr.predict(self.X)

        self.assertIsNotNone(predict_return, f"predict() method should return predicted values, but got: {type(predict_return)}")

        self.assertEqual(len(predict_return), len(self.X), f"predict() method should return predictions of length {len(self.X)}, but got: {len(predict_return)}")


class B_TestCostFunction(unittest.TestCase):
    def test_010_cost_grad_lambda_0(self):
        from hw4 import gradient, cost

        rng = np.random.RandomState(0)
        y = rng.randn(10)
        X = rng.randn(10, 5)


        _, cols = X.shape
        theta0 = np.ones(cols)

        grad = gradient(X, y, theta0, 0)

        def cost_(theta):
            return cost(X, y, theta, 0)

        eps = 10 ** -4
        theta0_ = theta0
        grad_num = np.zeros(grad.shape)
        for i in range(grad.size):
            theta0_[i] += eps
            h = cost_(theta0_)
            theta0_[i] -= 2 * eps
            l = cost_(theta0_)
            theta0_[i] += eps
            grad_num[i] = (h - l) / (2 * eps)

        np.testing.assert_almost_equal(grad, grad_num, decimal=4)

    def test_020_cost_grad_lambda_1(self):
        from hw4 import gradient, cost

        rng = np.random.RandomState(0)
        y = rng.randn(10)
        X = rng.randn(10, 5)


        _, cols = X.shape
        theta0 = np.ones(cols)

        grad = gradient(X, y, theta0, 1)

        def cost_(theta):
            return cost(X, y, theta, 1)

        eps = 10 ** -4
        theta0_ = theta0
        grad_num = np.zeros(grad.shape)
        for i in range(grad.size):
            theta0_[i] += eps
            h = cost_(theta0_)
            theta0_[i] -= 2 * eps
            l = cost_(theta0_)
            theta0_[i] += eps
            grad_num[i] = (h - l) / (2 * eps)

        np.testing.assert_almost_equal(grad, grad_num, decimal=4)


class C_TestLinearRegressoin(unittest.TestCase):
    def setUp(self) -> None:

        self.X = np.array(
            [[0.7, 0.9], 
             [0.1, 0.7], 
             [0.2, 0.8], 
             [0.0, 0.1], 
             [0.5, 0.0], 
             [0.6, 0.6]]
        )
        self.y = np.array(
            [7.5, 6.5, 6.8, 5.2, 5.5, 6.8]
        )
        
        self.intercept = 5.0
        self.coefs = np.array([1.0, 2.0])

        self.X_test = np.array([[0.8, 0.5], [0.3, 0.2], [0.9, 0.3], [0.4, 0.4]])
        
        # without regularization
        self.y_test = np.array([6.8, 5.7, 6.5, 6.2])

        # with regularization
        self.y_test_reg = {
            1: np.array([6.54893014, 6.08570555, 6.41364697, 6.30108098]),
            10: np.array([6.40968794, 6.33450745, 6.38725879, 6.36971714]),
        }
        self.coefs_reg = {
            1: np.array([0.40046129, 0.87664647]),
            10: np.array([0.06390273, 0.1440971]),
        }
        self.intercept_reg = {1: 5.790237870282883, 10: 6.286517209960804}

    def test_010_regularized_intercept(self):
        from hw4 import LinearRegression

        lr = LinearRegression(1)

        lr.fit(self.X, self.y)

        if lr.intercept < self.intercept:
            raise ValueError(
                f"Check your implementation. Seems like your intercept is regularized. Think about how to remove it from regularization."
            )

    def test_020_GD_no_regularization_correct_fit(self):
        from hw4 import LinearRegression

        lr = LinearRegression(0)
        lr.fit(self.X, self.y)

        fit_coefs = lr.coefs
        fit_intercept = lr.intercept


        np.testing.assert_almost_equal(fit_coefs, self.coefs, decimal=4, 
                                       err_msg="Gradient seem to produce different results than expected If close, try adjusting the threshold for convergence.")

        np.testing.assert_almost_equal(fit_intercept, self.intercept, decimal=4)

    def test_021_GD_no_regularization_correct_predict(self):
        from hw4 import LinearRegression

        lr = LinearRegression(0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)

        np.testing.assert_almost_equal(y_pred, self.y_test, decimal=4)

    def test_030_regularization_1_correct_fit(self):
        from hw4 import LinearRegression

        lr = LinearRegression(1)
        lr.fit(self.X, self.y)

        fit_coefs = lr.coefs
        fit_intercept = lr.intercept


        np.testing.assert_almost_equal(fit_coefs, self.coefs_reg[1], decimal=4,
                           err_msg="Regularized Gradient seem to produce different results than expected. If close, try adjusting the threshold for convergence or check your gradient for errors.")

        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[1], decimal=4)


    def test_031_regularization_1_correct_prediction(self):
        from hw4 import LinearRegression

        lr = LinearRegression(1)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)
        np.testing.assert_almost_equal(y_pred, self.y_test_reg[1], decimal=4)


    def test_040_regularization_10_correct_fit(self):
        from hw4 import LinearRegression

        lr = LinearRegression(10.0)
        lr.fit(self.X, self.y)

        fit_coefs = lr.coefs
        fit_intercept = lr.intercept

        np.testing.assert_almost_equal(fit_coefs, self.coefs_reg[10], decimal=4,
                                       err_msg="Regularized Gradient seem to produce different results than expected. If close, try adjusting the threshold for convergence or check your gradient for errors.")
        np.testing.assert_almost_equal(fit_intercept, self.intercept_reg[10], decimal=4)

    def test_041_regularization_10_correct_prediction(self):
        from hw4 import LinearRegression

        lr = LinearRegression(10.0)
        lr.fit(self.X, self.y)

        y_pred = lr.predict(self.X_test)
        np.testing.assert_almost_equal(y_pred, self.y_test_reg[10], decimal=4)



class D_TestVectorizedImplementation(unittest.TestCase):
    def test_010_vectorized(self):
        from hw4 import LinearRegression, cost, gradient

        self.assertFalse(
            uses_loop(cost), "Implementation of cost function is not vectorized."
        )

        self.assertFalse(
            uses_loop(gradient), "Implementation of gradient is not vectorized."
        )

        self.assertFalse(
            uses_loop(LinearRegression),
            "Methods in LR class should not have loops.",
        )

    def test_020_runtime(self):
        from hw4 import LinearRegression

        rng = np.random.RandomState(0)
        num_of_samples = 1_000
        num_of_features = 500
        y = rng.randn(num_of_samples)
        X = rng.randn(num_of_samples, num_of_features)

        timeout = 15

        start = time.time()
        lr = LinearRegression(0)
        lr.fit(X, y)
        end = time.time()

        self.assertLess(end - start, timeout, "Time taken to fit the model is too long.")


if __name__ == "__main__":
    unittest.main()