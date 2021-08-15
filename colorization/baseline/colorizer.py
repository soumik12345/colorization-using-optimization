import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import sparse
from typing import Tuple
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve

from .utils import position_to_id, find_neighbour


class Colorizer:

    def __init__(self, gray_image_file: str, visual_clues_file: str) -> None:
        self.original_gray_image = self.gray_image = np.array(Image.open(gray_image_file))
        self.original_visual_clues = self.visual_clues = np.array(Image.open(visual_clues_file))
    
    def _preprocess(self):
        self.gray_image = cv2.cvtColor(
            self.gray_image, cv2.COLOR_RGB2YUV) / 255.0
        self.visual_clues = cv2.cvtColor(
            self.visual_clues, cv2.COLOR_RGB2YUV) / 255.0
    
    def plot_inputs(self, figure_size: Tuple[int, int] = (12, 12)) -> None:
        figure = plt.figure(figsize=figure_size)
        figure.add_subplot(1, 2, 1).set_title('Black & White')
        plt.imshow(self.original_gray_image)
        plt.axis('off')
        figure.add_subplot(1, 2, 2).set_title('Color Hints')
        plt.imshow(self.original_visual_clues)
        plt.axis('off')
        plt.show()
    
    def plot_results(self, result: np.ndarray) -> None:
        fig = plt.figure(figsize=(25, 17))
        fig.add_subplot(1, 3, 1).set_title('Black & White')
        plt.imshow(self.original_gray_image)
        plt.axis('off')
        fig.add_subplot(1, 3, 2).set_title('Color Hints')
        plt.imshow(self.original_visual_clues)
        plt.axis('off')
        fig.add_subplot(1, 3, 3).set_title('Colorized')
        plt.imshow(result)
        plt.axis('off')
        plt.show()
    
    def colorize(self) -> np.ndarray:
        self._preprocess()
        n, m = self.gray_image.shape[0], self.gray_image.shape[1]
        size = n * m
        W = sparse.lil_matrix((size, size), dtype = float)
        b1 = np.zeros(shape = (size))
        b2 = np.zeros(shape = (size))
        for i in tqdm(range(n)):
            for j in range(m):
                if self.visual_clues[i, j, 0] > 1 - 1e-3:
                    id = position_to_id(i, j, m)
                    W[id, id] = 1
                    b1[id] = self.gray_image[i, j, 1]
                    b2[id] = self.gray_image[i, j, 2]
                    continue
                if abs(
                    self.gray_image[i, j, 0] - self.visual_clues[i, j, 0]
                ) > 1e-2 or abs(
                    self.gray_image[i, j, 1] - self.gray_image[i, j, 1]
                ) > 1e-2 or abs(
                    self.gray_image[i, j, 2] - self.visual_clues[i, j, 2]
                ) > 1e-2:
                    id = position_to_id(i, j, m)
                    W[id, id] = 1
                    b1[id] = self.visual_clues[i, j, 1]
                    b2[id] = self.visual_clues[i, j, 2]
                    continue
                Y = self.gray_image[i, j, 0]
                id = position_to_id(i, j, m)
                neighbour = find_neighbour(i, j, n, m)
                Ys, ids, weights = [], [], []
                for pos in neighbour:
                    Ys.append(self.gray_image[pos[0], pos[1], 0])
                    ids.append(position_to_id(pos[0], pos[1], m))
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
                W[id, id] += 1.
        result = np.zeros(shape = (n, m, 3))
        result[:, :, 0] = self.gray_image[:, :, 0]
        W = W.tocsc()
        u = spsolve(W, b1)
        v = spsolve(W, b2)
        for i in range(n):
            for j in range(m):
                id = position_to_id(i, j, m)
                result[i, j, 1], result[i, j, 2] = u[id], v[id]
        result = (np.clip(result, 0., 1.) * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_YUV2RGB)
        return result
