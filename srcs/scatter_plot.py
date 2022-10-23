import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkgs.feature import Feature
from pkgs.parsing import check_input


def create_dict(houses) -> dict:
	courses = dict()
	for house in houses:
		courses[house] = dict()
		for name, dtype in houses[house].dtypes.iteritems():
			if dtype == np.float64:
				column = [float(x) for x in houses[house][name].values if not math.isnan(x)]
				courses[house][name] = Feature(name, column)
	return courses


def plot(houses):
	for house in houses:
		plt.scatter(houses[house]["Defense Against the Dark Arts"], houses[house]["Astronomy"], alpha=0.4)

	plt.xlabel("Defense Against the Dark Arts")
	plt.ylabel("Astronomy")
	plt.legend(houses.keys())

	plt.show()


def main():
	check_input(sys.argv)

	data = pd.read_csv(sys.argv[1], index_col=0)
	houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}

	courses = create_dict(houses)
	plot(houses)


if __name__ == '__main__':
	main()
