import pandas as pd
import sys
import numpy as np
import math
import os
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
	print(f'Please provide one parameter with the csv file.')
	quit()

filename, extension = os.path.splitext(sys.argv[1])
if extension != '.csv' or not os.path.exists(sys.argv[1]):
	print(f'Please provide a valid .csv file.')
	quit()

data = pd.read_csv(sys.argv[1], index_col=0)
houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}
features = list()


class Feature:
	def __init__(self, n, col):
		self.name = n
		self.count = len(col)
		self.mean = float(sum(col) / len(col))
		col.sort()
		self.min, self.max = col[0], col[-1]
		self.p25, self.p50, self.p75 = col[int(len(col) / 4)], col[int(len(col) / 2)], col[int(len(col) / 4 * 3)]
		self.std = math.sqrt(sum([float((float(x) - self.mean) ** 2) for x in col]) / self.count)

	def getvalue(self, val):
		return {
			'': self.name,
			'Count': self.count,
			'Mean': self.mean,
			'Std': self.std,
			'25%': self.p25,
			'50%': self.p50,
			'75%': self.p75,
			'Min': self.min,
			'Max': self.max
		}[val]


courses = {}
course_list = set()
for house in houses:
	courses[house] = dict()
	for name, dtype in houses[house].dtypes.iteritems():
		if dtype == np.float64:
			course_list.add(name)
			column = [float(x) for x in houses[house][name].values if not math.isnan(x)]
			courses[house][name] = Feature(name, column)

for house in houses:
	plt.scatter(houses[house]["Defense Against the Dark Arts"], houses[house]["Astronomy"])

plt.xlabel("Defense Against the Dark Arts")
plt.ylabel("Astronomy")
plt.legend(houses.keys())

plt.show()
