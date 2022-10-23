import numpy as np


def accepts(*types):
    def check_accepts(f):
        if len(types) != f.__code__.co_argcount:
            return None

        def new_f(*args, **kwargs):
            if any(not isinstance(arg, t) for arg, t in zip(args, types)):
                return None
            return f(*args, **kwargs)

        # new_f.__name__ = f.__name__
        return new_f

    return check_accepts


class MyLogisticRegression:
    """
    Description: My personal logistic regression to classify things.
    If a function has the __ prefix, it means that it assumes the x value has a column of ones already...
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.005, max_iter: int = 10_000):
        if not isinstance(thetas, np.ndarray) or not isinstance(alpha, float) or not isinstance(max_iter, int):
            raise TypeError('Bad arguments given to MyLogisticRegression')
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    @accepts(np.ndarray)
    def sigmoid_(x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        return 1 / (1 + np.exp(-x))

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        ones = np.ones(shape=(x.shape[0], 1))
        x = np.hstack((ones, x))
        y_hat = self.__predict_(x)
        return x.T.dot(y_hat - y) / y.shape[0]

    def __gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_hat = self.__predict_(x)
        return x.T.dot(y_hat - y) / y.shape[0]

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        :param x: np.ndarray
        :param y: np.ndarray
        :return: new theta values
        """
        x = np.column_stack((np.ones(shape=(x.shape[0], 1)), x))
        for idx in range(self.max_iter):
            assert not np.isnan(self.__gradient_(x, y)).any()
            self.thetas -= (self.alpha * self.__gradient_(x, y))
        return self.thetas

    @staticmethod
    @accepts(np.ndarray, np.ndarray, float)
    def loss_elem_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> np.ndarray | None:
        """
        :param y: Actual values as an np.ndarray
        :param y_hat: Predicted values as an np.ndarray
        :param eps: very small value
        :return: np.ndarray of the losses
        """
        if y.shape != y_hat.shape:
            return None
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    @staticmethod
    @accepts(np.ndarray, np.ndarray, float)
    def loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        loss_elem = MyLogisticRegression.loss_elem_(y, y_hat, eps)
        return -loss_elem.sum() / y.shape[0]

    def __predict_(self, x: np.ndarray) -> np.ndarray:
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        assert not np.isnan(x.dot(self.thetas)).any()
        return MyLogisticRegression.sigmoid_(x.dot(self.thetas))

    def predict_(self, x: np.ndarray) -> np.ndarray | None:
        """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        ones = np.ones(shape=(x.shape[0], 1))
        x = np.hstack((ones, x))
        return MyLogisticRegression.sigmoid_(x.dot(self.thetas))
