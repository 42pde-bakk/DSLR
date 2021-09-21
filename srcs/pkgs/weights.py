import numpy as np
from .logisticregression import LogisticRegression


def save_weights(lr: LogisticRegression, filename: str) -> None:
	with open(filename, 'w') as f:
		f.write(','.join(lr.target_uniques) + '\n')
		f.write('biases:\n' + ','.join([str(bias[0]) for bias in lr.biases]) + '\nweights:\n')
		for feature in range(lr.weights.shape[1]):
			row = ','.join([str(lr.weights[target][feature]) for target in range(lr.weights.shape[0])])
			f.write(row + '\n')


def load_weights(filename: str) -> LogisticRegression:
	lr = LogisticRegression()
	with open(filename, 'r') as f:
		lines = f.read().split()
		lr.target_uniques = lines[0].strip().split(sep=',')
		lr.biases = np.zeros((len(lr.target_uniques), 1))
		biases = [float(bias) for bias in lines[2].split(sep=',')]
		lr.biases = np.fromiter(biases, dtype=np.float64).reshape(-1, 1)
		lr.weights = np.zeros((len(lr.target_uniques), len(lines) - 4))
		features = {i: list() for i, _ in enumerate(lr.target_uniques)}
		for row in range(4, len(lines)):
			row_weights = lines[row].split(sep=',')
			for i, _ in enumerate(lr.target_uniques):
				features[i].append(row_weights[i])
		for i, _ in enumerate(lr.target_uniques):
			lr.weights[i] = np.fromiter(features[i], dtype=float)
	return lr
