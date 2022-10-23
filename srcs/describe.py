import math
import sys

import numpy as np
import pandas as pd

from pkgs.feature import Feature
from pkgs.parsing import check_input


def print_formatted(features):
	longest_name = max([len(feature.getvalue('')) for feature in features])
	for row in ["", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
		print(row.ljust(longest_name), end=' ')
		for f in features:
			print(str(f.getvalue(row)).ljust(longest_name), end=' ')
		print()


def main():
	check_input(sys.argv)

	data = pd.read_csv(sys.argv[1])
	features = set()

	for name, dtype in data.dtypes.iteritems():
		if dtype == np.float64 and name != 'Hogwarts House':
			# House check is because if Hogwarts House is empty, it's also seen as a float64
			column = [float(x) for x in data[name].values if not math.isnan(x)]
			features.add(Feature(name, column))
	print_formatted(features)


if __name__ == "__main__":
	main()
