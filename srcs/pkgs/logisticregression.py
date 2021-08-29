import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification

# Source: https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2


class LogisticRegression:
    def __init__(self, batch_size=50000, epochs=100, learning_rate=0.0005):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def loss(y, y_hat):
        # y is 0 or 1, y_hat is a number between 0 and 1
        return -np.mean(y * np.log(y_hat)) - (1 - y) * np.log(1 - y_hat)

    @staticmethod
    def normalize(X):
        # X --> Input.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        return X

    def gradients(self, X, y, y_hat):
        # X --> Input.
        # y --> true/target value.
        # y_hat --> hypothesis/predictions.
        # w --> weights (parameter).
        # b --> bias (parameter).

        # m -> number of training examples
        m = X.shape[0]

        # Gradient of loss w.r.t weights.
        dw = (1 / m) * np.dot(X.T, (y_hat - y))

        # Gradient of loss w.r.t bias.
        db = (1 / m) * np.sum((y_hat - y))

        return dw, db

    # def plot_decision_boundary(self, X, w, b, y):
    #     # X --> Inputs
    #     # w --> weights
    #     # b --> bias
    #
    #     # The Line is y=mx+c
    #     # So, Equate mx+c = w.X + b
    #     # Solving we find m and c
    #     x1 = [min(X[:, 0]), max(X[:, 0])]
    #     m = -w[0] / w[1]
    #     c = -b / w[1]
    #     x2 = m * x1 + c
    #
    #     # Plotting
    #     fig = plt.figure(figsize=(10, 8))
    #     plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "g^")
    #     plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    #     plt.xlim([-2, 2])
    #     plt.ylim([0, 2.2])
    #     plt.xlabel("feature 1")
    #     plt.ylabel("feature 2")
    #     plt.title('Decision Boundary')
    #     plt.plot(x1, x2, 'y-')
    #     plt.show()

    def fit(self, X, y):
        # X --> Input.
        # y --> true/target value.
        # bs --> Batch Size.
        # epochs --> Number of iterations.
        # lr --> Learning rate.

        # m-> number of training examples
        # n-> number of features
        m, n = X.shape

        # Initializing weights and bias to zeros.
        w = np.zeros((n, 1))
        b = 0

        # Reshaping y
        y = y.reshape(m, 1)

        # Normalizing the inputs
        x = self.normalize(X)

        # Empty list to store losses
        losses = list()

        # Training loop.
        for epoc in range(self.epochs):
            for i in range((m - 1) // self.batch_size + 1):
                # Defining batches. SGD.
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]

                # Calculating hypothesis/prediction
                y_hat = self.sigmoid(np.dot(xb, w) + b)

                # Getting the gradients of loss w.r.t. parameters
                dw, db = self.gradients(xb, yb, y_hat)

                # Updating the paramters
                w -= self.learning_rate * dw
                b -= self.learning_rate * db

            # Calculating loss and appending it to the list.
            losses.append(self.loss(y, self.sigmoid(np.dot(X, w) + b)))
        # Returning weights, bias and losses(List).
        return w, b, losses

    def predict(self, X: np.array):
        # X --> Input

        # Normalizing the inputs.
        x = self.normalize(X)
        preds = self.sigmoid(np.dot(X, w) + b)

        # If y_hat >= 0.5 --> round up to 1
        pred_class = [1 if i >= 0.5 else 0 for i in preds]
        return np.array(pred_class)

    def accuracy(self, y, y_hat):
        return np.sum(y == y_hat) / len(y)


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    mylr = LogisticRegression(np.array([2, 0.5, 7.1, -4.3, 2.09]))

    print(mylr.predict(X))
    print(mylr.cost(X,Y))
    mylr.fit(X,Y)
    print(mylr.theta)
    print(mylr.cost(X,Y))
