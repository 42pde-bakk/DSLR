from pkgs.parsing import predict_check_input
import pandas as pd
import numpy as np
import sys
import pickle

predict_check_input(sys.argv)

df = pd.read_csv(sys.argv[1], index_col=0)

df.fillna(method='ffill', inplace=True)
X = np.array(df.values[:, np.arange(7, 11)], dtype=float)  # Hogwarts course score to predict Hogwarts house
y = df.values[:, 0]  # Hogwarts House

LogReg = pickle.load(open('datasets/weights', 'rb'))

y_pred = LogReg.predict(X)

with open('datasets/houses.csv', 'w+') as f:
	f.write('Index,Hogwarts House\n')
	for i in range(len(y_pred)):
		f.write(f'{i}, {y_pred[i]}\n')
