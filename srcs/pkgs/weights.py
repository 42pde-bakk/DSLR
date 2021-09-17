import pickle


def save_weights(weights) -> None:
	return pickle.dump(weights, open('datasets/weights', 'wb'))


def load_weights(filename: str):
	return pickle.load(open(filename, 'rb'))
