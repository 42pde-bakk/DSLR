import copy
import pickle

from pkgs.parsing import check_input
from pkgs.my_logistic_regression import MyLogisticRegression as MyLogR
import pandas as pd
import numpy as np
import sys

FEATURES = [
	'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
	'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
	'Flying'
]

TARGET_COLUMN = 'Hogwarts House'
OUTPUT_FILENAME = 'houses.csv'


def run_predictions():
	check_input(sys.argv, argv_len=3)
	csv_file = sys.argv[1]
	models_file = sys.argv[2]

	df = pd.read_csv(csv_file, index_col=0)
	df.fillna(0.0, inplace=True)

	x = df[FEATURES].to_numpy().reshape(-1, len(FEATURES))
	x = (x - x.mean(axis=0)) / x.std(axis=0)  # normalize the data

	with open(models_file, 'rb') as f:
		models = pickle.load(f)

	y_hat = MyLogR.multiclass_predict_(models, x)
	unique_houses = models[0].unique_outcomes
	data = []
	for idx, pred in enumerate(y_hat):
		data.append([unique_houses[pred[0]]])
	df = pd.DataFrame(data, columns=['Hogwarts House'])
	df.index.name = 'Index'
	df.to_csv(OUTPUT_FILENAME)


if __name__ == '__main__':
	run_predictions()
