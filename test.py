import sys
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def pos2id(x, y, n, m):
	return x * m + y

def id2pos(id, n, m):
	return id // m, id % m

def nearby(x, y, n, m, d = 2):
	neighbour = []
	for i in range(max(0, x - d), min(n, x + d + 1)):
		for j in range(max(0, y - d), min(m, y + d + 1)):
			if (i != x) or (j != y):
				neighbour.append([i, j])
	return neighbour

def colorize(gray, sketch):
	n, m = gray.shape[0], gray.shape[1]
	size = n * m
	cnt = 0
	W = sparse.lil_matrix((size, size), dtype = float)
	b1 = np.zeros(shape = (size))
	b2 = np.zeros(shape = (size))
	
	for i in range(n):
		for j in range(m):
			if sketch[i, j, 0] > 1 - 1e-3:
				id = pos2id(i, j, n, m)
				W[id, id] = 1
				b1[id] = gray[i, j, 1]
				b2[id] = gray[i, j, 2]
				continue
			if abs(gray[i, j, 0] - sketch[i, j, 0]) > 1e-2 or abs(gray[i, j, 1] - sketch[i, j, 1]) > 1e-2 or abs(gray[i, j, 2] - sketch[i, j, 2]) > 1e-2:
				id = pos2id(i, j, n, m)
				W[id, id] = 1
				b1[id] = sketch[i, j, 1]
				b2[id] = sketch[i, j, 2]
				cnt += 1
				continue
			Y = gray[i, j, 0]
			id = pos2id(i, j, n, m)
			neighbour = nearby(i, j, n, m)
			Ys = []
			ids = []
			weights = []
			for pos in neighbour:
				Ys.append(gray[pos[0], pos[1], 0])
				ids.append(pos2id(pos[0], pos[1], n, m))
			sigma = np.std(Ys)
			sum = 0.
			for k in range(len(neighbour)):
				if sigma > 1e-3:
					w = np.exp(-1 * (Ys[k] - Y) * (Ys[k] - Y) / 2 / sigma / sigma)
					sum += w
					weights.append(w)
				else:
					sum += 1.
					weights.append(1.)
			for k in range(len(neighbour)):
				weights[k] /= sum
				W[id, ids[k]] += -1 * weights[k]
				'''W[ids[k], id] += -1 * weights[k]
				W[ids[k], ids[k]] += weights[k] * weights[k]
				for l in range(k):
					W[ids[k], ids[l]] += weights[k] * weights[l]
					W[ids[l], ids[k]] += weights[k] * weights[l]'''
			W[id, id] += 1.
	
	print(cnt)
	output = np.zeros(shape = (n, m, 3))
	output[:, :, 0] = gray[:, :, 0]
				
	W = W.tocsc()
	#print(W)
	
	u = spsolve(W, b1)
	v = spsolve(W, b2)
	for i in range(n):
		for j in range(m):
			id = pos2id(i, j, n, m)
			output[i, j, 1], output[i, j, 2] = u[id], v[id]
	return output

if __name__ == "__main__":
	try:
		gray_dir = sys.argv[1]
		sketch_dir = sys.argv[2]
		output_dir = sys.argv[3]
	except:
		print("Usage: colorize <gray> <sketch> <output>")
		exit(0)
	
	try:
		gray = cv2.imread(gray_dir)
		sketch = cv2.imread(sketch_dir)
	except:
		print("Failed to read images.")
		exit(0)
	
	assert gray.shape == sketch.shape, "The two images should share the same size."
	
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2YUV)
	sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2YUV)
	
	gray = gray / 255.0
	sketch = sketch / 255.0
	
	output = colorize(gray, sketch)
	output = (np.clip(output, 0., 1.) * 255).astype(np.uint8)
	
	output = cv2.cvtColor(output, cv2.COLOR_YUV2BGR)
	cv2.imwrite(output_dir, output)