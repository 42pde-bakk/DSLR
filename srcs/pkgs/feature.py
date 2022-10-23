import math


def variance(items: list) -> float:
	mean = sum(items) / len(items)
	return sum([float((float(x) - mean) ** 2) for x in items]) / len(items)


def std_deviation(items: list) -> float:
	return math.sqrt(variance(items))


def mean_absolute_deviation(items: list) -> float:
	mean = sum(items) / len(items)
	return sum(abs(item - mean) for item in items) / len(items)


class Feature:
	def __init__(self, n: str, col: list) -> None:
		self.name = n
		self.count = len(col)
		self.mean = float(sum(col) / len(col))
		col.sort()
		self.min, self.max = col[0], col[-1]
		self.p25, self.p50, self.p75 = col[int(len(col) / 4)], col[int(len(col) / 2)], col[int(len(col) / 4 * 3)]
		self.std = std_deviation(col)
		self.var = variance(col)
		self.mad = mean_absolute_deviation(col)

	def getvalue(self, val: str):
		match val:
			case '':
				return self.name
			case 'Count':
				return self.count
			case 'Mean':
				return self.mean
			case 'Std':
				return self.std
			case 'Var':
				return self.var
			case '25%':
				return self.p25
			case '50%':
				return self.p50
			case '75%':
				return self.p75
			case 'Min':
				return self.min
			case 'Max':
				return self.max
			case 'Mad':
				return self.mad
