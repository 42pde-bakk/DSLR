import copy

from pkgs.parsing import check_input
from pkgs.my_logistic_regression import MyLogisticRegression as MyLogR
from pkgs.data_splitter import data_splitter
import pandas as pd
import numpy as np
import sys
import pickle

FEATURES = [
	'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
	'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
	'Flying'
]

TARGET_COLUMN = 'Hogwarts House'


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
		model = MyLogR(thetas, alpha=0.0001, max_iter=20_000)
		model.set_params(unique_outcomes=unique_houses)
		new_train_y = np.where(train_y == i, 1, 0)
		model.fit_(train_x, new_train_y)
		models.append(copy.deepcopy(model))
	accuracy = MyLogR.combine_models(models, test_x, test_y)
	print(f'accuracy = {accuracy:.1f}%')

	with open('models.pickle', 'wb') as f:
		pickle.dump(models, f)


if __name__ == '__main__':
	run_training()
