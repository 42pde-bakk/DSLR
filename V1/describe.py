from pkgs.parsing import check_input
from pkgs.feature import Feature
import pandas as pd
import sys
import numpy as np
import math

check_input(sys.argv)

data = pd.read_csv(sys.argv[1])
features = list()

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
