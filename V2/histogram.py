from pkgs.parsing import check_input
from pkgs.feature import Feature
import pandas as pd
import sys
import numpy as np
import math
import os
import matplotlib.pyplot as plt


def lowest_std_std(course_list, courses, houses) -> str:
	ret = str()
	double_std = math.inf
	for i, c in enumerate(course_list):
		stds = [courses[h][c].getvalue('Std') for h in houses]
		mean = sum(stds) / len(stds)
		std_of_stds = math.sqrt(sum([(std - mean) ** 2 for std in stds]))
		if std_of_stds < double_std:
			ret = c
			double_std = std_of_stds
	return ret


def show_histogram(houses, lowest):
	for house in houses:
		plt.hist(houses[house][lowest], density=True, label=house, bins=30, alpha=0.5)
	plt.title(lowest)
	plt.show()


def main():
	check_input(sys.argv)

	data = pd.read_csv(sys.argv[1], index_col=0)
	houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}

	courses = dict()
	course_list = set()
	for house in houses:
		courses[house] = dict()
		for name, dtype in houses[house].dtypes.iteritems():
			if dtype == np.float64:
				course_list.add(name)
				column = [float(x) for x in houses[house][name].values if not math.isnan(x)]
				courses[house][name] = Feature(name, column)

	lowest = lowest_std_std(course_list, courses, houses)
	show_histogram(houses, lowest)


if __name__ == "__main__":
	main()

