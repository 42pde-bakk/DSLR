import numpy as np

# Source: https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# Multinomial logistic regression:
# https://towardsdatascience.com/ml-from-scratch-multinomial-logistic-regression-6dda9cbacf9d


class LogisticRegression:
	def __init__(self, n_iterations=2000, learning_rate=1e-5, normalization: bool = True):
		self.n_iterations = n_iterations
		self.learning_rate = learning_rate
		self.thetas = (0, 0)
		self.weights = np.ndarray
		self.biases = np.ndarray
		self.target_uniques = list()
		self.normalization = normalization

	@staticmethod
	def sigmoid(z):
		return 1.0 / (1 + np.exp(-z))

	@staticmethod
	def normalize(x):
		m, n = x.shape
		for i in range(n):
			x = (x - x.mean(axis=0)) / x.std(axis=0)
		return x

	@staticmethod
	def cross_entropy_loss(probabilities, target):
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
	def train_test_split(x, y, test_size=0.2):
		"""Splits dataset into training and testing sets.

		Args-
			dataframe- The dataframe object you want to split
			test_size- Size of test dataset that you want

		Returns-
			train_features, train_target, test_features, test_target
		"""

		test_rows = np.round(len(x) * test_size)
		random_rows = np.random.randint(0, len(x), int(test_rows))
		test_x = np.array([x[i] for i in random_rows])
		test_y = np.array([y[i] for i in random_rows])

		train_x = np.delete(x, random_rows, axis=0)
		train_y = np.delete(y, random_rows, axis=0)
		return train_x, train_y, test_x, test_y

	def initial_weights(self, features, target):
		feature_amount, target_options = features.shape[1], len(np.unique(target))
		self.weights = np.zeros((target_options, feature_amount))
		self.biases = np.zeros((target_options, 1))

	def linear_predict(self, feature_matrix):
		"""This is the linear predictor function for out MLR model. It calculates the logit scores for each possible outcome.

		Args-
			featureMat- A numpy array of features
			weights- A numpy array of weights for our model
			biases- A numpy array of biases for our model

		Returns-
			logit_scores- Logit scores for each possible outcome of the target variable
			for each feature set in the feature matrix
		"""
		# creating empty(garbage value) array for each feature set
		logit_scores = np.array([np.empty([len(self.biases)]) for _ in range(feature_matrix.shape[0])])

		for i in range(feature_matrix.shape[0]):
			# calculates logit score for each feature set then flattens the logit vector
			logit_scores[i] = (np.dot(self.weights, feature_matrix[i].reshape(-1, 1)) + self.biases).reshape(-1)
		return logit_scores

	def softmax_normalizer(self, logit_matrix):
		"""Converts logit scores for each possible outcome to probability values.

		Args-
			logitMatrix - This is the output of our logitPredict function; consists  logit scores for each feature set

		Returns-
			probabilities - Probability value of each outcome for each feature set
		"""

		probabilities = np.array([np.empty([len(self.biases)]) for _ in range(
			logit_matrix.shape[0])])  # creating empty(garbage value) array for each feature set
		for i in range(logit_matrix.shape[0]):
			exp = np.exp(logit_matrix[i])  # exponentiates each element of the logit array
			sum_of_arr = np.sum(exp)  # adds up all the values in the exponentiated array
			probabilities[i] = exp / sum_of_arr  # logit scores to probability values
		return probabilities

	def multinomial_logistic_regression(self, features):
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
		logit_scores = self.linear_predict(features)
		probabilities = self.softmax_normalizer(logit_scores)
		predictions = np.array([self.target_uniques[np.argmax(i)] for i in probabilities])  # returns the outcome with max probability
		return probabilities, predictions

	def predict(self, x_test):
		if self.normalization:
			x_test = self.normalize(x_test)
		probabilities, predictions = self.multinomial_logistic_regression(x_test)
		return predictions

	def stochastic_gradient_descent(self, features, target):
		"""Performs stochastic gradient descent optimization on the model.

		Args-
			learning_rate- Size of the step the function will take during optimization
			target- Numpy array containing actual target values
			features- Numpy array of independent variables

		Returns-
			loss_list- Array containing list of losses observed after each epoch
		"""
		self.initial_weights(features, target)
		loss_list = np.array([])

		for i in range(self.n_iterations):
			# calculates possibilities for each possible outcome
			probabilities, _ = self.multinomial_logistic_regression(features)

			# Calculates cross entropy loss for actual target and predictions
			ce_loss = self.cross_entropy_loss(probabilities, target)
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

	def fit(self, x_train, y_train, solver='sgd'):
		self.target_uniques = np.unique(y_train).tolist()
		if self.normalization:
			x_train = self.normalize(x_train)
		y_train = np.unique(y_train, return_inverse=True)[1]  # All target values mapped to values from 0 to (n-1)
		return self.stochastic_gradient_descent(x_train, y_train)

	@staticmethod
	def accuracy(predictions, target):
		"""Calculates total accuracy for our model.

		Args-
			predictions- Predicted target outcomes as predicted by our MLR function
			target- Actual target values

		Returns-
			accuracy- Accuracy percentage of our model
		"""
		correct_preds = sum([pred == actual for pred, actual in zip(predictions, target)])
		accuracy = correct_preds / len(predictions) * 100
		print(f'Predicted {correct_preds}/{len(predictions)} correct, that\'s {accuracy:.2f}%')
		return accuracy
