from pydataset import data
import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


def print_result():
	correct = 0
	for test, pred in zip(y_test, y_pred):
		if test == pred:
			correct += 1
	print(f'Got {correct}/{len(y_test)} correct!')


if len(sys.argv) != 2:
	print(f'Please provide one parameter with the csv file.')
	quit()

filename, extension = os.path.splitext(sys.argv[1])
if extension != '.csv' or not os.path.exists(sys.argv[1]):
	print(f'Please provide a valid .csv file.')
	quit()

df = pd.read_csv(sys.argv[1], index_col=0)
df.dropna(subset=['Defense Against the Dark Arts', 'Charms', 'Herbology', 'Divination', 'Muggle Studies'], inplace=True)

X = np.array(df.values[:, [7, 8, 9]], dtype=float)
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
print(y_pred)

print_result()

pickle.dump(LogReg, open('datasets/weights', 'wb'))
