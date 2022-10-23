import copy
import pickle
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pkgs.data_splitter import data_splitter
from pkgs.my_logistic_regression import MyLogisticRegression as MyLogR
from pkgs.parsing import check_input

FEATURES = [
	'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
	'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
	'Flying'
]

TARGET_COLUMN = 'Hogwarts House'


def plot_predictions(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> None:
	fig, axs = plt.subplots(nrows=2, ncols=2)
	for i in range(4):
		feature = FEATURES[i]
		plot = axs[i // 2, i % 2]
		plot.set_title(feature)
		x_col = x[:, i]
		size = 15
		plot.scatter(x_col, y, label='True houses', s=5 * size)
		plot.scatter(x_col, y_hat, label='My algo\'s predictions', s=2 * size)
		plot.legend(loc='best')

	plt.show()


def run_training():
	check_input(sys.argv)
	csv_file = sys.argv[1]

	df = pd.read_csv(csv_file, index_col=0)
	df.fillna(0.0, inplace=True)

	y = df[TARGET_COLUMN].to_numpy().reshape(-1, 1)
	x = df[FEATURES].to_numpy().reshape(-1, len(FEATURES))
	x = (x - x.mean(axis=0)) / x.std(axis=0)  # normalize the data
	unique_houses = np.unique(y)
	houses_dict = {house: idx for idx, house in enumerate(unique_houses)}
	for k, v in houses_dict.items():
		y[y == k] = v
	train_x, test_x, train_y, test_y = data_splitter(x, y, 0.8)
	models = []

	for i, house in enumerate(unique_houses):
		# Train a model for each house (One vs All)
		print(f'Let\'s train model {i} for {house}')
		thetas = np.ones(shape=(len(FEATURES) + 1, 1))
		model = MyLogR(thetas, alpha=0.001, max_iter=100, gd_type='Stochastic')
		model.set_params(unique_outcomes=unique_houses)
		new_train_y = np.where(train_y == i, 1, 0)
		model.fit_(train_x, new_train_y)
		models.append(copy.deepcopy(model))
	accuracy = MyLogR.combine_models(models, test_x, test_y)
	print(f'accuracy = {accuracy:.1f}%')

	with open('models.pickle', 'wb') as f:
		pickle.dump(models, f)

	y_hat = MyLogR.multiclass_predict_(models, test_x)
	plot_predictions(test_x, y_hat, test_y)


if __name__ == '__main__':
	run_training()
