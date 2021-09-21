from pkgs.parsing import predict_check_input
from pkgs.logisticregression import LogisticRegression
import pandas as pd
import numpy as np
import sys
from pkgs.weights import load_weights


def write_to_csv(y_pred: np.ndarray, predictions_filename: str):
	with open(predictions_filename, 'w+') as f:
		f.write('Index,Hogwarts House\n')
		for i in range(len(y_pred)):
			# You can't have a space after the comma.
			# evaluate.py just splits on the comma and sees every char after it as your prediction.
			# And 'Gryffindor' != ' Gryffindor'
			f.write(f'{i},{y_pred[i]}\n')


def main():
	predict_check_input(sys.argv)

	df = pd.read_csv(sys.argv[1], index_col=0)
	df.drop(labels=['First Name', 'Last Name', 'Birthday', 'Best Hand'], inplace=True, axis=1)
	df.drop('Hogwarts House', inplace=True, axis=1)
	df.fillna(0, inplace=True)  # Fill all NaN's with 0's
	x = df.to_numpy()

	lr = load_weights('datasets/weights.csv')

	y_pred = lr.predict(x)
	write_to_csv(y_pred, 'datasets/houses.csv')


if __name__ == '__main__':
	main()
