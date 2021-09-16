from pkgs.parsing import check_input
from pkgs.logisticregression import LogisticRegression
import pandas as pd
import numpy as np
import sys
import pickle


def print_result(y_test, y_pred):
	correct = 0
	for test, pred in zip(y_test, y_pred):
		if test == pred:
			correct += 1
	print(f'Got {correct}/{len(y_test)} correct!')


def save_weights(logreg):
	pickle.dump(logreg, open('datasets/weights', 'wb'))


def parse_data():
	col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']  # column names

	df = pd.read_csv('datasets/processed.cleveland.data.csv', header=None)
	df.columns = col_names  # setting dataframe column names
	df.replace({'?': np.nan}, inplace=True)
	df[['ca', 'thal']] = df[['ca', 'thal']].astype('float64')  # Casting columns data-type to floats
	df['ca'].replace({np.nan: df['ca'].median()}, inplace=True)  # replaces null values of ca column with median value
	df['thal'].replace({np.nan: df['thal'].median()}, inplace=True)
	return df


def split_dataset(df):
	# selecting all the features within our dataset
	features = df[
		['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
	]
	features = features.to_numpy()  # converts feature set to numpy array
	target = df['num'].to_numpy()  # converts target column to numpy array
	print(features.shape, len(target))
	return features, target


def test(df):
	lr = LogisticRegression()
	x_train, y_train, x_test, y_test = lr.train_test_split(df)
	# print(f'shapes:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	loss_list = lr.stochastic_gradient_descent(x_train, y_train)
	testProbs, testPreds = lr.multinomialLogReg(x_test)
	acc = lr.accuracy(testPreds, y_test)
	print(acc)
	return acc


def mymain(count=1):
	df = parse_data()
	total = 0
	for i in range(count):
		total += test(df)
	print(f'average = {total / count}')

	# features, target = split_dataset(df)
	# lr = LogisticRegression()
	# standardized_features = LogisticRegression.standardize_data(features)
	# probs, preds = lr.multinomialLogReg(standardized_features, target)
	# print(lr.accuracy(preds, target))


if __name__ == '__main__':
	mymain()
