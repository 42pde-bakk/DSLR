import numpy as np
from .logisticregression import LogisticRegression


def save_weights(lr: LogisticRegression, filename: str) -> None:
	with open('datasets/test', 'w') as f:
		print(lr.biases[0].shape)
		f.write(str(lr.target_uniques) + '\n')
		f.write(str(lr.biases) + str(lr.biases.shape) + '\n')
		f.write(str(lr.weights) + str(lr.weights.shape) + '\n')
	with open(filename, 'w') as f:
		f.write(','.join(lr.target_uniques) + '\n')
		f.write('biases:\n' + ','.join([str(bias[0]) for bias in lr.biases]) + '\nweights:\n')
		for feature in range(lr.weights.shape[1]):
			row = ','.join([str(lr.weights[target][feature]) for target in range(lr.weights.shape[0])])
			f.write(row + '\n')


def load_weights(filename: str):
	lr = LogisticRegression()
	with open(filename, 'r') as f:
		lines = f.read().split()
		lr.target_uniques = lines[0].strip().split(sep=',')
		lr.biases = np.zeros((len(lr.target_uniques), 1))
		biases = [float(bias) for bias in lines[2].split(sep=',')]
		lr.biases = np.fromiter(biases, dtype=float).reshape(-1, 1)
		print(lr.biases)
		# lr.weights = np.zeros((len(lr.target_uniques), len(lines) - 4))
		# for i in range(4, len(lines)):

	return lr
