import math
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkgs.feature import Feature, std_deviation
from pkgs.parsing import check_input


def lowest_std_std(course_list: set, courses: dict, houses: dict) -> Tuple[pd.DataFrame, str]:
	ret, double_std = str(), math.inf
	df = pd.DataFrame(columns=houses.keys())
	df.insert(0, 'Course', 'empty')
	df['Average'] = 100

	for i, c in enumerate(course_list):
		stds = [courses[h][c].getvalue('Std') for h in houses]
		std_of_stds = std_deviation(stds)
		if std_of_stds < double_std:
			ret = c
			double_std = std_of_stds
		row = {h: courses[h][c].getvalue('Std') for h in houses}
		row['Course'] = c
		row['Average'] = std_of_stds
		df = df.append(row, ignore_index=True)

	df = df.sort_values(by='Average')
	return df, ret


def show_histogram(houses: dict, lowest: str) -> None:
	for house, color in zip(houses, ['red', 'blue', 'orange', 'black']):
		plt.hist(houses[house][lowest], density=True, label=house, bins=30, alpha=0.35, color=color)
	plt.title(lowest)
	plt.legend(houses.keys())
	plt.show()


def create_courses_dict(houses: dict):
	courses, course_list = dict(), set()
	for house in houses:
		courses[house] = dict()
		for name, dtype in houses[house].dtypes.iteritems():
			if dtype == np.float64:
				course_list.add(name)
				column = [float(x) for x in houses[house][name].values if not math.isnan(x)]
				courses[house][name] = Feature(name, column)
	return courses, course_list


def main():
	check_input(sys.argv)
	data = pd.read_csv(sys.argv[1], index_col=0)
	houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}
	courses, course_list = create_courses_dict(houses)
	df, lowest = lowest_std_std(course_list, courses, houses)
	print(df.to_string(index=False))
	show_histogram(houses, lowest)


if __name__ == "__main__":
	main()
