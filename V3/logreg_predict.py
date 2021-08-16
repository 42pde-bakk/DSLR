from pkgs.parsing import predict_check_input
import pandas as pd
import numpy as np
import sys
import pickle


def write_to_csv(y_pred):
	with open('datasets/houses.csv', 'w+') as f:
		f.write('Index,Hogwarts House\n')
		for i in range(len(y_pred)):
			# You can't have a space after the comma.
			# evaluate.py just splits on the comma and sees every char after it as your prediction.
			# And 'Gryffindor' != ' Gryffindor'
			f.write(f'{i},{y_pred[i]}\n')


def main():
	predict_check_input(sys.argv)

	df = pd.read_csv(sys.argv[1], index_col=0)

	df.fillna(method='ffill', inplace=True)
	x = np.array(df.values[:, np.arange(7, 11)], dtype=float)  # Hogwarts course score to predict Hogwarts house

	log_reg = pickle.load(open('datasets/weights', 'rb'))

	y_pred = log_reg.predict(x)

	write_to_csv(y_pred)


if __name__ == '__main__':
	main()
