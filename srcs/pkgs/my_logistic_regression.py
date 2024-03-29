from __future__ import annotations

import enum

import numpy as np

from .other_metrics import accuracy_score_


class GDType(enum.Enum):
	DEFAULT = 0,
	STOCHASTIC = 1,
	BATCH = 2,
	MINI_BATCH = 3

	@classmethod
	def parse_str(cls, s: str) -> GDType:
		match s:
			case 'Default':
				return cls.DEFAULT
			case 'Stochastic':
				return cls.DEFAULT
			case 'Batch':
				return cls.DEFAULT
			case 'Mini-Batch':
				return cls.DEFAULT
		return NotImplemented


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

	def __init__(self, thetas: np.ndarray, alpha: float = 0.005, max_iter: int = 10_000, gd_type: str = 'Standard'):
		if not isinstance(thetas, np.ndarray) or not isinstance(alpha, float) or not isinstance(max_iter, int):
			raise TypeError('Bad arguments given to MyLogisticRegression')
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.unique_outcomes = list()
		self.gd_type = GDType.parse_str(gd_type)

	def get_params(self) -> dict:
		"""Get parameters for this estimator."""
		return vars(self)

	def set_params(self, **params) -> MyLogisticRegression:
		"""Set the parameters of this estimator."""
		for key, value in params.items():
			if key in vars(self).keys():
				setattr(self, key, value)
		return self

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

	def __fit_stochastically_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		for idx in range(self.max_iter):
			if self.gd_type == GDType.STOCHASTIC:
				rng_state = np.random.get_state()
				np.random.shuffle(x)
				np.random.set_state(rng_state)  # To ensure the arrays are still in unison
				np.random.shuffle(y)
				for row_idx in range(x.shape[0]):
					self.thetas -= (self.alpha * self.__gradient_(x[row_idx], y[row_idx]))
			return self.thetas

	def __fit_mini_batch_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		# Default implementation (Batch Gradient Descent)
		batch_amount = 10
		batches_x = np.split(x, batch_amount)
		batches_y = np.split(y, batch_amount)

		for _ in range(self.max_iter):
			for xbatch, ybatch in zip(batches_x, batches_y):
				self.thetas -= (self.alpha * self.__gradient_(xbatch, ybatch))
		return self.thetas

	def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""
		:param x: np.ndarray
		:param y: np.ndarray
		:return: new theta values
		"""
		x = np.column_stack((np.ones(shape=(x.shape[0], 1)), x))
		if self.gd_type == GDType.STOCHASTIC:
			return self.__fit_stochastically_(x, y)
		elif self.gd_type == GDType.MINI_BATCH:
			return self.__fit_mini_batch_(x, y)

		# Default implementation (Batch Gradient Descent)
		for idx in range(self.max_iter):
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

	@staticmethod
	def multiclass_predict_(models: list[MyLogisticRegression], x_test: np.ndarray) -> np.ndarray:
		predict_together = np.hstack([m.predict_(x_test) for m in models])
		return predict_together.argmax(axis=1).reshape(-1, 1)

	@staticmethod
	def combine_models(models: list[MyLogisticRegression], x_test: np.ndarray, y_test: np.ndarray) -> float:
		"""
		Combines N models to finalize our one-vs-all predictions
		:param models: list of our N models
		:param x_test: numpy array of our input
		:param y_test: numpy array of the expected output
		:return: a float value between 0 and 1 showcasing how accurate our combined model is
		"""
		y_hat = MyLogisticRegression.multiclass_predict_(models, x_test)
		accuracy = accuracy_score_(y_test, y_hat) * 100
		return accuracy
