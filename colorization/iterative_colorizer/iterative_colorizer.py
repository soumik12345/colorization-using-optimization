import cv2
import scipy
import colorsys
import numpy as np
from tqdm import tqdm
from typing import Tuple
from skimage.io import imread
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve

from .window_neighbour import WindowNeighbor
from .utils import affinity_a, to_seq


class IterativeColorizer:

    def __init__(self, original_image: str, visual_clues: str) -> None:
        self.image_oiginal_rgb = cv2.imread(original_image)
        self.image_original = self.image_oiginal_rgb.astype(float) / 255
        self.image_clues_rgb = imread(visual_clues)
        self.image_clues = self.image_clues_rgb.astype(float) / 255
    
    def plot_inputs(self, figure_size: Tuple[int, int] = (12, 12)) -> None:
        figure = plt.figure(figsize=figure_size)
        figure.add_subplot(1, 2, 1).set_title('Black & White')
        plt.imshow(self.image_original)
        plt.axis('off')
        figure.add_subplot(1, 2, 2).set_title('Color Hints')
        plt.imshow(self.image_clues)
        plt.axis('off')
        plt.show()
    
    def plot_results(self, result: np.ndarray) -> None:
        fig = plt.figure(figsize=(25, 17))
        fig.add_subplot(1, 3, 1).set_title('Black & White')
        plt.imshow(self.image_original)
        plt.axis('off')
        fig.add_subplot(1, 3, 2).set_title('Color Hints')
        plt.imshow(self.image_clues)
        plt.axis('off')
        fig.add_subplot(1, 3, 3).set_title('Colorized')
        plt.imshow(result)
        plt.axis('off')
        plt.show()
    
    def yuv_channels_to_rgb(self, channel_y, channel_u, channel_v) -> np.ndarray:
        """Combine 3 channels of YUV to a RGB photo: n x n x 3 array"""
        result_rgb = [colorsys.yiq_to_rgb(
            channel_y[i], channel_u[i], channel_v[i]
        ) for i in range(len(self.result_y))]
        result_rgb = np.array(result_rgb)
        image_rgb = np.zeros(self.image_yuv.shape)
        image_rgb[:, :, 0] = result_rgb[:, 0].reshape(self.image_rows, self.image_cols, order='F')
        image_rgb[:, :, 1] = result_rgb[:, 1].reshape(self.image_rows, self.image_cols, order='F')
        image_rgb[:, :, 2] = result_rgb[:, 2].reshape(self.picimage_rows_rows, self.image_cols, order='F')
        return image_rgb
    
    def colorize(self) -> np.ndarray:
        (self.image_rows, self.image_cols, _) = self.image_original.shape
        image_size = self.image_rows * self.image_cols
        channel_Y, _, _ = colorsys.rgb_to_yiq(
            self.image_original[:, :, 0],
            self.image_original[:, :, 1],
            self.image_original[:, :, 2]
        )
        _, channel_U, channel_V = colorsys.rgb_to_yiq(
            self.image_clues[:, :, 0],
            self.image_clues[:, :, 1],
            self.image_clues[:, :, 2]
        )
        map_colored = (abs(channel_U) + abs(channel_V)) > 0.0001
        self.image_yuv = np.dstack((channel_Y, channel_U, channel_V))
        weight_data = []
        # num_pixel_bw = 0
        wd_width = 1
        progress_bar = tqdm(range(self.image_cols))
        progress_bar.set_description('Finding neighbouring pixels...')
        for c in progress_bar:
            for r in range(self.image_rows):
                window_neighbour = WindowNeighbor(wd_width, (r, c), self.image_yuv)
                if not map_colored[r,c]:
                    weights = affinity_a(window_neighbour)
                    for e in weights:
                        weight_data.append([window_neighbour.center, (e[0], e[1]), e[2]])
                weight_data.append([
                    window_neighbour.center, (window_neighbour.center[0], window_neighbour.center[1]), 1.])
        sparse_index_data = [
            [
                to_seq(e[0][0], e[0][1], self.image_rows),
                to_seq(e[1][0], e[1][1], self.image_rows), e[2]
            ] for e in weight_data
        ]
        sparse_index_row_col = np.array(sparse_index_data, dtype=np.integer)[:, 0:2]
        sparse_data = np.array(sparse_index_data, dtype=np.float64)[:, 2]
        weight_matrix = scipy.sparse.csr_matrix(
            (sparse_data, (sparse_index_row_col[:,0], sparse_index_row_col[:,1])),
            shape=(image_size, image_size)
        )
        b_u = np.zeros(image_size)
        b_v = np.zeros(image_size)
        idx_colored = np.nonzero(map_colored.reshape(image_size, order='F'))
        pic_u_flat = self.image_yuv[:,:,1].reshape(image_size, order='F')
        b_u[idx_colored] = pic_u_flat[idx_colored]
        pic_v_flat = self.image_yuv[:,:,2].reshape(image_size, order='F')
        b_v[idx_colored] = pic_v_flat[idx_colored]
        self.result_y = self.image_yuv[:,:,0].reshape(image_size, order='F')
        self.result_u = spsolve(weight_matrix, b_u)
        self.result_v = spsolve(weight_matrix, b_v)
        result = self.yuv_channels_to_rgb(self.result_y, self.result_u, self.result_v)
        return result
