from pkgs.parsing import check_input
from pkgs.logisticregression import LogisticRegression
from pkgs.weights import save_weights
import pandas as pd
import sys


def harrypotter():
	check_input(sys.argv)

	df = pd.read_csv(sys.argv[1], index_col=0)
	df.drop(labels=['First Name', 'Last Name', 'Birthday', 'Best Hand'], inplace=True, axis=1)
	df.fillna(0, inplace=True)  # Fill all NaN's with 0's

	y = df['Hogwarts House'].to_numpy()
	df.drop('Hogwarts House', inplace=True, axis=1)
	x = df.to_numpy()

	lr = LogisticRegression(n_iterations=100)
	train_x, train_y, test_x, test_y = lr.train_test_split(x, y, test_size=0.2)
	lr.fit(train_x, train_y)
	test_preds = lr.predict(test_x)
	acc = lr.accuracy(test_preds, test_y)
	save_weights(lr, 'datasets/weights.csv')
	return acc


if __name__ == '__main__':
	harrypotter()
