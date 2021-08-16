import cv2
import scipy
import colorsys
import numpy as np
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
        ) for i in range(len(self.ansY))]
        result_rgb = np.array(result_rgb)
        image_rgb = np.zeros(self.pic_yuv.shape)
        image_rgb[:, :, 0] = result_rgb[:, 0].reshape(self.pic_rows, self.pic_cols, order='F')
        image_rgb[:, :, 1] = result_rgb[:, 1].reshape(self.pic_rows, self.pic_cols, order='F')
        image_rgb[:, :, 2] = result_rgb[:, 2].reshape(self.pic_rows, self.pic_cols, order='F')
        return image_rgb
    
    def colorize(self) -> np.ndarray:
        (self.pic_rows, self.pic_cols, _) = self.image_original.shape
        pic_size = self.pic_rows * self.pic_cols
        channel_Y,_,_ = colorsys.rgb_to_yiq(
            self.image_original[:, :, 0],
            self.image_original[:, :, 1],
            self.image_original[:, :, 2]
        )
        _,channel_U,channel_V = colorsys.rgb_to_yiq(
            self.image_clues[:, :, 0],
            self.image_clues[:, :, 1],
            self.image_clues[:, :, 2]
        )
        map_colored = (abs(channel_U) + abs(channel_V)) > 0.0001
        self.pic_yuv = np.dstack((channel_Y, channel_U, channel_V))
        weight_data = []
        # num_pixel_bw = 0
        wd_width = 1
        for c in range(self.pic_cols):
            for r in range(self.pic_rows):
                res = []
                w = WindowNeighbor(wd_width, (r, c), self.pic_yuv)
                if not map_colored[r,c]:
                    weights = affinity_a(w)
                    for e in weights:
                        weight_data.append([w.center,(e[0],e[1]), e[2]])
                weight_data.append([w.center, (w.center[0],w.center[1]), 1.])
        sp_idx_rc_data = [
            [
                to_seq(e[0][0], e[0][1], self.pic_rows),
                to_seq(e[1][0], e[1][1], self.pic_rows), e[2]
            ] for e in weight_data
        ]
        sp_idx_rc = np.array(sp_idx_rc_data, dtype=np.integer)[:, 0:2]
        sp_data = np.array(sp_idx_rc_data, dtype=np.float64)[:, 2]
        matA = scipy.sparse.csr_matrix(
            (sp_data, (sp_idx_rc[:,0], sp_idx_rc[:,1])),
            shape=(pic_size, pic_size)
        )
        b_u = np.zeros(pic_size)
        b_v = np.zeros(pic_size)
        idx_colored = np.nonzero(map_colored.reshape(pic_size, order='F'))
        pic_u_flat = self.pic_yuv[:,:,1].reshape(pic_size, order='F')
        b_u[idx_colored] = pic_u_flat[idx_colored]
        pic_v_flat = self.pic_yuv[:,:,2].reshape(pic_size, order='F')
        b_v[idx_colored] = pic_v_flat[idx_colored]
        self.ansY = self.pic_yuv[:,:,0].reshape(pic_size, order='F')
        ansU = spsolve(matA, b_u)
        ansV = spsolve(matA, b_v)
        result = self.yuv_channels_to_rgb(self.ansY, ansU, ansV)
        return result
