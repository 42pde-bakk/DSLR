import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Source: https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2

# Multinomial logistic regression:
# https://towardsdatascience.com/ml-from-scratch-multinomial-logistic-regression-6dda9cbacf9d


class LogisticRegression:
	def __init__(self, batch_size=32, n_iterations=2000, learning_rate=1e-5):
		self.batch_size = batch_size
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.thetas = (0, 0)
		self.weights = np.ndarray
		self.biases = np.ndarray

	@staticmethod
	def sigmoid(z):
		return 1.0 / (1 + np.exp(-z))

	@staticmethod
	def normalize(x):
		m, n = x.shape
		for i in range(n):
			x = (x - x.mean(axis=0)) / x.std(axis=0)
		return x


	def crossEntropyLoss(self, probabilities, target):
		"""Calculates cross entropy loss for a set of predictions and actual targets.

		Args-
			predictions- Probability predictions, as returned by multinomialLogReg function
			target- Actual target values
		Returns-
			ce_loss- Average cross entropy loss
		"""
		n_samples = probabilities.shape[0]
		ce_loss = 0
		for sample, i in zip(probabilities, target):
			ce_loss += -np.log(sample[i])
		ce_loss /= n_samples
		return ce_loss

	@staticmethod
	def train_test_split(df, test_size=0.2, normalize_data: bool = True):
		"""Splits dataset into training and testing sets.

		Args-
			dataframe- The dataframe object you want to split
			test_size- Size of test dataset that you want

		Returns-
			train_features, train_target, test_features, test_target
		"""
		data = df.to_numpy()
		totalRows = data.shape[0]
		testRows = np.round(totalRows * test_size)
		randRowNum = np.random.randint(0, int(totalRows), int(testRows))  # randomly generated row numbers
		testData = np.array([data[i] for i in randRowNum])
		data = np.delete(data, randRowNum, axis=0)

		train_features, train_target = data[:, :-1], data[:, -1]
		test_features, test_target = testData[:, :-1], testData[:, -1]
		if normalize_data:
			train_features = LogisticRegression.normalize(train_features)
			test_features = LogisticRegression.normalize(test_features)
		return train_features, train_target, test_features, test_target

	def initial_weights(self, features, target):
		feature_amount, target_options = features.shape[1], len(np.unique(target))
		self.weights = np.random.rand(target_options, feature_amount)
		self.biases = np.random.rand(target_options, 1)

	def linear_predict(self, feature_matrix):
		"""This is the linear predictor function for out MLR model. It calculates the logit scores for each possible outcome.

		Args-
			featureMat- A numpy array of features
			weights- A numpy array of weights for our model
			biases- A numpy array of biases for our model

		Returns-
			logit_scores- Logit scores for each possible outcome of the target variable for each feature set in the feature matrix
		"""
		# creating empty(garbage value) array for each feature set
		logit_scores = np.array([np.empty([5]) for _ in range(feature_matrix.shape[0])])

		for i in range(feature_matrix.shape[0]):
			# calculates logit score for each feature set then flattens the logit vector
			logit_scores[i] = (self.weights.dot(feature_matrix[i].reshape(-1, 1)) + self.biases).reshape(-1)
		return logit_scores

	def softmax_normalizer(self, logit_matrix):
		"""Converts logit scores for each possible outcome to probability values.

		Args-
			logitMatrix - This is the output of our logitPredict function; consists  logit scores for each feature set

		Returns-
			probabilities - Probability value of each outcome for each feature set
		"""

		probabilities = np.array([np.empty([5]) for i in range(
			logit_matrix.shape[0])])  # creating empty(garbage value) array for each feature set
		for i in range(logit_matrix.shape[0]):
			exp = np.exp(logit_matrix[i])  # exponentiates each element of the logit array
			sumOfArr = np.sum(exp)  # adds up all the values in the exponentiated array
			probabilities[i] = exp / sumOfArr  # logit scores to probability values
		return probabilities

	def multinomialLogReg(self, features):
		"""Performs logistic regression on a given feature set.

		Args-
			features- Numpy array of features(standardized)
			weights- A numpy array of weights for our model
			biases- A numpy array of biases for our model

		Returns-
			probabilities, predictions
			Here,
				probabilities: Probability values for each possible outcome for each feature set in the feature matrix
				predictions: Outcome with max probability for each feature set
		"""
		logitScores = self.linear_predict(features)
		probabilities = self.softmax_normalizer(logitScores)
		predictions = np.array([np.argmax(i) for i in probabilities])  # returns the outcome with max probability
		return probabilities, predictions

	def stochastic_gradient_descent(self, features, target):
		"""Performs stochastic gradient descent optimization on the model.

		Args-
			learning_rate- Size of the step the function will take during optimization
			epochs- No. of iterations the function will run for on the model
			target- Numpy array containing actual target values
			features- Numpy array of independent variables
			weights- Numpy array containing weights associated with each feature
			biases- Array containinig model biases

		Returns-
			weights, biases, loss_list
			where,
				weights- Latest weight calculated (Numpy array)
				bias- Latest bias calculated (Numpy array)
				loss_list- Array containing list of losses observed after each epoch
		"""
		self.initial_weights(features, target)
		target = target.astype(int)
		loss_list = np.array([])

		for i in range(self.n_iterations):
			# calculates possibilities for each possible outcome
			probabilities, _ = self.multinomialLogReg(features)

			# Calculates cross entropy loss for actual target and predictions
			ce_loss = self.crossEntropyLoss(probabilities, target)
			# Adds the ce_loss value for the epoch to loss_list
			loss_list = np.append(loss_list, ce_loss)

			# Subtract 1 from the scores of the correct outcome
			probabilities[np.arange(features.shape[0]), target] -= 1

			grad_weight = probabilities.T.dot(features)  # Gradient of loss w.r.t. weights
			grad_biases = np.sum(probabilities, axis=0).reshape(-1, 1)  # Gradient of loss w.r.t biases

			# Updating weights and biases
			self.weights -= (self.learning_rate * grad_weight)
			self.biases -= (self.learning_rate * grad_biases)

		return loss_list

	@staticmethod
	def accuracy(predictions, target):
		"""Calculates total accuracy for our model.

		Args-
			predictions- Predicted target outcomes as predicted by our MLR function
			target- Actual target values

		Returns-
			accuracy- Accuracy percentage of our model
		"""
		correct_preds = 0
		for i in range(len(predictions)):
			if predictions[i] == target[i]:
				correct_preds += 1
		accuracy = correct_preds / len(predictions) * 100
		return accuracy
