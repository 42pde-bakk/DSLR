import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkgs.feature import Feature
from pkgs.parsing import check_input


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


def start_plotting(houses, course_list):
	plt.close('all')
	nrows, ncols = 3, 5
	colours = {
		'Gryffindor': 'red',
		'Ravenclaw': 'blue',
		'Slytherin': 'green',
		'Hufflepuff': 'gold'
	}

	figure, axes = plt.subplots(nrows=nrows, ncols=ncols)
	axs = axes.flatten()
	for i, course in enumerate(course_list):
		plot = axs[i]
		plot.set_title(course)
		for house_name, house_value in houses.items():
			plot.scatter(range(len(house_value[course])), house_value[course], marker='.', color=colours[house_name], label=house_name, alpha=0.8)

	plt.show()


def main():
	check_input(sys.argv)

	data = pd.read_csv(sys.argv[1], index_col=0)
	houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}

	courses, course_list = create_dict(houses)
	# History of Magic and Transfiguration are very similar
	start_plotting(houses, course_list)


if __name__ == '__main__':
	main()
