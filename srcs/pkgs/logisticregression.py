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
	def __init__(self, batch_size=32, n_iterations=50000, learning_rate=1e-5):
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
	def standardization_check(features):
		# checking if standardization worked
		total_cols = features.shape[1]  # total number of columns
		for i in range(total_cols):
			print(features[:, i].std())

	@staticmethod
	def standardize_data(feature_array):
		"""Takes the numpy.ndarray object containing the features and performs standardization on the matrix.
		The function iterates through each column and performs scaling on them individually.

		Args-
			feature_array- Numpy array containing training features
		"""
		total_cols = feature_array.shape[1]
		for i in range(total_cols):
			feature_col = feature_array[:, i]
			mean = feature_col.mean()
			std = feature_col.std()
			feature_array[:, i] = (feature_array[:, i] - mean) / std  # Standard scaling of each element in the column
		return feature_array

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
		print(feature_matrix.shape[0])
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

	def multinomialLogReg(self, features, target):
		self.initial_weights(features, target)
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

	def accuracy(self, predictions, target):
		"""Calculates total accuracy for our model.

		Args-
			predictions- Predicted target outcomes as predicted by our MLR function
			target- Actual target values

		Returns-
			accuracy- Accuracy percentage of our model
		"""
		correctPred = 0
		for i in range(len(predictions)):
			if predictions[i] == target[i]:
				correctPred += 1
		accuracy = correctPred / len(predictions) * 100
		return accuracy