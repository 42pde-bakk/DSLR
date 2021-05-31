import pandas as pd
import sys
import numpy as np
import math
import os

if len(sys.argv) != 2:
	print(f'Please provide one parameter with the csv file.')
	quit()

filename, extension = os.path.splitext(sys.argv[1])
if extension != '.csv' or not os.path.exists(sys.argv[1]):
	print(f'Please provide a valid .csv file.')
	quit()

data = pd.read_csv(sys.argv[1])
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


for name, dtype in data.dtypes.iteritems():
	if dtype == np.float64 and name != 'Hogwarts House':
		column = [float(x) for x in data[name].values if not math.isnan(x)]
		features.append(Feature(name, column))

longest_name = max([len(feature.getvalue('')) for feature in features])

rows = ["", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

for row in rows:
	print(row.ljust(longest_name), end=' ')
	for f in features:
		print(str(f.getvalue(row)).ljust(longest_name), end=' ')
	print("")
