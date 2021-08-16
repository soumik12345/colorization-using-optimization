def position_to_id(x, y, m):
	return x * m + y


def find_neighbour(x, y, n, m, d = 2):
	neighbour = []
	for i in range(max(0, x - d), min(n, x + d + 1)):
		for j in range(max(0, y - d), min(m, y + d + 1)):
			if (i != x) or (j != y):
				neighbour.append([i, j])
	return neighbour
