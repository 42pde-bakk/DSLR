from pkgs.parsing import check_input
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


def print_result():
	correct = 0
	for test, pred in zip(y_test, y_pred):
		if test == pred:
			correct += 1
	print(f'Got {correct}/{len(y_test)} correct!')


check_input(sys.argv)

df = pd.read_csv(sys.argv[1], index_col=0)
df.fillna(0, inplace=True)  # Fill all NaN's with 0's

X = np.array(df.values[:, np.arange(7, 11)], dtype=float)  # Course score to train the model on
y = df.values[:, 0]   # Hogwarts House

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

print_result()

pickle.dump(LogReg, open('datasets/weights', 'wb'))
