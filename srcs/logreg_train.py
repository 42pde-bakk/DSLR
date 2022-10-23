import copy

from pkgs.parsing import check_input
from pkgs.my_logistic_regression import MyLogisticRegression as MyLogR
from pkgs.data_splitter import data_splitter
from pkgs.other_metrics import accuracy_score_, f1_score_
from pkgs.weights import save_weights
import pandas as pd
import numpy as np
import sys

FEATURES = [
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
    'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
    'Flying'
]

TARGET_COLUMN = 'Hogwarts House'
# FEATURES = [
# 	'Care of Magical Creatures', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts'
# ]


def combine_models(models: list[MyLogR], x_test: np.ndarray, y_test: np.ndarray, houses_dict: dict):
    predict_together = np.hstack([m.predict_(x_test) for m in models])
    # for i in range(len(models)):
    # 	print(predict_together[i][0])
    y_hat = predict_together.argmax(axis=1).reshape(-1, 1)
    # u, inv = np.unique(y_hat, return_inverse=True)
    print(f'accuracy = {accuracy_score_(y_test, y_hat) * 100:.1f}%')


def harrypotter():
    check_input(sys.argv)

    df = pd.read_csv(sys.argv[1], index_col=0)
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
        model = MyLogR(thetas=np.ones(shape=(len(FEATURES) + 1, 1)),
                       alpha=0.0001, max_iter=20_000)
        new_train_y = np.where(train_y == i, 1, 0)
        model.fit_(train_x, new_train_y)
        models.append(copy.deepcopy(model))
    combine_models(models, test_x, test_y, houses_dict)


if __name__ == '__main__':
    harrypotter()
