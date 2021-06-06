from pkgs.parsing import check_input
from pkgs.feature import Feature
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_dict(houses) -> tuple:
	courses = {}
	course_list = set()
	for house in houses:
		courses[house] = dict()
		for name, dtype in houses[house].dtypes.iteritems():
			if dtype == np.float64:
				course_list.add(name)
				column = [float(x) for x in houses[house][name].values if not math.isnan(x)]
				courses[house][name] = Feature(name, column)
	return courses, course_list


def plot(houses, course_list):
	plt.close('all')

	figure, axes = plt.subplots(len(course_list), len(course_list), figsize=(24, 14))
	for row_plt, row_course in zip(axes, course_list):
		for col_plt, col_course in zip(row_plt, course_list):
			for house in houses:
				if row_course != col_course:
					col_plt.scatter(houses[house][col_course], houses[house][row_course], alpha=0.4)
				else:
					col_plt.hist(houses[house][row_course], density=True, label=house, bins=30, alpha=0.5)
			col_plt.tick_params(labelbottom=False, labelleft=False)

			if col_plt.get_subplotspec().is_first_col():
				col_plt.set_ylabel(row_course.replace(' ', '\n'))

			if col_plt.get_subplotspec().is_last_row():
				col_plt.set_xlabel(col_course.replace(' ', '\n'))

	plt.legend(houses.keys())

	plt.show()


def main():
	check_input(sys.argv)

	data = pd.read_csv(sys.argv[1], index_col=0)
	houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}

	courses, course_list = create_dict(houses)
	plot(houses, course_list)


if __name__ == '__main__':
	main()
