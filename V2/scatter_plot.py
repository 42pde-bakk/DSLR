from pkgs.parsing import check_input
from pkgs.feature import Feature
import pandas as pd
import sys
import numpy as np
import math
import os
import matplotlib.pyplot as plt

check_input(sys.argv)

data = pd.read_csv(sys.argv[1], index_col=0)
houses = {x: pd.DataFrame(y) for x, y in data.groupby('Hogwarts House', as_index=False)}

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
	plt.scatter(houses[house]["Defense Against the Dark Arts"], houses[house]["Astronomy"], alpha=0.4)

plt.xlabel("Defense Against the Dark Arts")
plt.ylabel("Astronomy")
plt.legend(houses.keys())

plt.show()
