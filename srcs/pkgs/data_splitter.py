import sys
from typing import Tuple

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


@accepts(np.ndarray, np.ndarray, float)
def data_splitter(x: np.ndarray, y: np.ndarray, proportion: float) -> Tuple | None:
	"""Splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.
	Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		proportion: has to be a float, the proportion of the dataset that will be assigned to the
		training set.
	Return:
		(x_train, x_test, y_train, y_test) as a tuple of numpy.array
		None if x or y is an empty numpy.array.
		None if x and y do not share compatible dimensions.
		None if x, y or proportion is not of expected type.
	Raises:
		This function should not raise any Exception.
	"""
	if x.shape[0] != y.shape[0]:
		print(f'Why are you giving me arrays of differing sizes?', file=sys.stderr)
		return

	cutoff = int(proportion * x.shape[0])
	x_train, x_test = np.split(x, [cutoff])
	y_train, t_test = np.split(y, [cutoff])
	return x_train, x_test, y_train, t_test
