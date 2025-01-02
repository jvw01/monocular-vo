import numpy as np
from scipy import signal
import math


def harris(img, patch_size, kappa):
    """ Returns the harris scores for an image given a patch size and a kappa value
        The returned scores are of the same shape as the input image """
    sobel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    
    I_x = signal.convolve2d(img, sobel, mode="valid") # matrix with I_x value for each pixel
    I_y = signal.convolve2d(img, sobel.T, mode="valid")
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y
    sum_I_x2 = signal.convolve2d(I_x2, np.ones([patch_size, patch_size]), mode="valid") # convolve I_x2 with a box filter to get sum of I_x2 in a patch
    sum_I_y2 = signal.convolve2d(I_y2, np.ones([patch_size, patch_size]), mode="valid")
    sum_I_xy = signal.convolve2d(I_xy, np.ones([patch_size, patch_size]), mode="valid")
    
    # print("path_size: " + str(patch_size))
    # print("h,w: " + str(img.shape[0]) + ", " + str(img.shape[1]))
    # print("I_x: " + str(I_x.shape[0]) + ", " + str(I_x.shape[1]))
    # print("I_y2: " + str(I_y2.shape[0]) + ", " + str(I_y2.shape[1]))
    # print("I_xy: " + str(I_xy.shape[0]) + ", " + str(I_xy.shape[1]))
    # print("sum_I_x2: " + str(sum_I_x2.shape[0]) + ", " + str(sum_I_x2.shape[1]))
    
    R_harris = (sum_I_x2 * sum_I_y2 - sum_I_xy ** 2) - kappa * (sum_I_x2 + sum_I_y2) ** 2
    # print("R_harris: " + str(R_harris.shape[0]) + ", " + str(R_harris.shape[1]))
    
    R_harris = np.pad(R_harris, pad_width=math.ceil(patch_size/2))
    # print("R_harris_padded: " + str(R_harris.shape[0]) + ", " + str(R_harris.shape[1]))

    R_harris = np.where(R_harris < 0, 0, R_harris)
    
    return R_harris
    