import math
from typing import List, Tuple, Union

import cv2
import numpy as np

from . import processing


def swf(
    in_arr: np.ndarray, ksize: int = None, filter_op: str = "mean"
) -> np.ndarray:
    """
    Method to apply a pixel-by-pixel median or mean filter using the side
    window technique.

    :param in_arr: 2D/3D array
    :param ksize: Window size (odd number)
    :param filter_op: Filter operation to be used (median/mean)
    :return: Filtered array
    """
    from numpy.lib.stride_tricks import as_strided

    # Available filter operations
    filter_dict = {"median": np.median, "mean": np.mean}
    filter_op = filter_dict[filter_op]

    # Get window radius, padded array and strides
    ksize += 1 - ksize % 2  # Make sure window size is an odd number
    radius = ksize // 2  # Window radius
    padded = np.pad(in_arr, radius, "reflect")  # Pad array using the radius
    strides = padded.strides + padded.strides

    # Parameters that depend on the input array
    try:
        assert in_arr.ndim == 2
        sy, sx = in_arr.shape
        bands = False
        reshape_shape = -1
        pr_axes = [2, 3]
    except AssertionError:
        sy, sx, bands = in_arr.shape
        reshape_shape = (1, sy * sx, bands)
        pr_axes = [3, 4]

    # Calculate output shape
    if not bands:
        up_down_shape = (sy + radius, sx, ksize - radius, ksize)
        left_right_shape = (sy, sx + radius, ksize, ksize - radius)
        others_shape = (sy + radius, sx + radius, ksize - radius, ksize - radius)
    else:
        up_down_shape = (sy + radius, sx, 1, ksize - radius, ksize, bands)
        left_right_shape = (sy, sx + radius, 1, ksize, ksize - radius, bands)
        others_shape = (
            sy + radius,
            sx + radius,
            1,
            ksize - radius,
            ksize - radius,
            bands,
        )

    # Slice the padded array using strides
    up_down = as_strided(padded, shape=up_down_shape, strides=strides)
    left_right = as_strided(padded, shape=left_right_shape, strides=strides)
    rest = as_strided(padded, shape=others_shape, strides=strides)
    padded = None

    # Get the median value of each sub-window, then flatten them
    up_down_meds = np.apply_over_axes(filter_op, up_down, pr_axes).astype(
        up_down.dtype
    )
    up_down = None
    left_right_meds = np.apply_over_axes(filter_op, left_right, pr_axes).astype(
        left_right.dtype
    )
    left_right = None
    rest_meds = np.apply_over_axes(filter_op, rest, pr_axes).astype(rest.dtype)
    rest = None

    # Compute filter for subwindows
    up_meds = up_down_meds[:-radius, :].reshape(reshape_shape)
    down_meds = up_down_meds[radius:, :].reshape(reshape_shape)
    left_meds = left_right_meds[:, :-radius].reshape(reshape_shape)
    right_meds = left_right_meds[:, radius:].reshape(reshape_shape)
    nw_meds = rest_meds[:-radius, :-radius].reshape(reshape_shape)
    sw_meds = rest_meds[radius:, :-radius].reshape(reshape_shape)
    ne_meds = rest_meds[:-radius, radius:].reshape(reshape_shape)
    se_meds = rest_meds[radius:, radius:].reshape(reshape_shape)

    # Stack the flattened arrays and find where the minimum value is for each pixel
    stacked = np.vstack(
        (
            up_meds,
            down_meds,
            right_meds,
            left_meds,
            nw_meds,
            sw_meds,
            ne_meds,
            se_meds,
        )
    )

    # remove from memory
    up_meds = None
    down_meds = None
    right_meds = None
    left_meds = None
    nw_meds = None
    sw_meds = None
    ne_meds = None
    se_meds = None

    subtr = np.absolute(stacked - in_arr.reshape(reshape_shape))
    in_arr = None
    inds = np.argmin(
        subtr, axis=0
    )  # Get indices where the subtr is minimum along the 0
    # axis
    subtr = None

    # Get the output pixel values
    if not bands:
        filt = np.take_along_axis(
            stacked, np.expand_dims(inds, axis=0), axis=0
        ).reshape(sy, sx)
    else:
        filt = np.take_along_axis(
            stacked, np.expand_dims(inds, axis=0), axis=0
        ).reshape(sy, sx, bands)

    stacked = None

    return filt

def kernel_disp(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Method to get the mean displacement within a tile.

    :param img1: Image1 tile
    :param img2: Image2 tile
    :return: Mean subpixel displacement
    """

    if np.all(img1 == 0) or np.all(img2 == 0):
        return 0

    # Phase correlation
    fft_img1 = np.fft.fft2(img1)  # FFT
    fft_img2 = np.fft.fft2(img2)
    conj_img2 = np.ma.conjugate(fft_img2)  # Complex conjugate
    R_12 = fft_img1 * conj_img2  # Cross-power spectrum
    R_12 /= np.absolute(R_12)

    disp_map_12 = np.fft.ifft2(R_12).real  # Normalized cross-correlation
    # Estimate mean displacement in pixels for each band pair

    mean_disp = np.mean(estimate_disp(disp_map_12))
    return mean_disp


def phase_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    ksize: int = 3,
    eq_histogram: bool = None,
    transf: Union[Tuple, List] = None,
    use_sliding_tiles: bool = False,
) -> Tuple[np.ndarray, Tuple]:
    """
    Wrapper method to estimate subpixel displacement between 2 grayscale
    images.

    :param img1: Grayscale image array
    :param img2: Grayscale image array
    :param ksize: Kernel size
    :param eq_histogram: Enable histogram equalization
    :param transf: Geo-transformation tuple
    :param use_sliding_tiles: Flag to enable using sliding windows instead 
    of tiles
    :return: Subpixel displacement map,  geo-transformation tuple
    """
    if img1.ndim > 2:
        raise ValueError("img1 is not grayscale (multiple bands were detected).")
    if img2.ndim > 2:
        raise ValueError("img2 is not grayscale (multiple bands were detected).")

    if transf is None:
        transf = (0, 1.0, 0, 0, 0, -1.0)

    # Histogram equalization to enhance results
    if eq_histogram:
        cv2.equalizeHist(img1, img1)
        cv2.equalizeHist(img2, img2)


    # Get tiles for each band
    if use_sliding_tiles:
        img1_wins = processing.get_sliding_win(in_arr=img1, ksize=ksize)
        img2_wins = processing.get_sliding_win(in_arr=img2, ksize=ksize)
        new_tr = transf
    else:
        img1_wins = processing.get_tiles(in_arr=img1, ksize=ksize)
        img2_wins = processing.get_tiles(in_arr=img2, ksize=ksize)
        
        # Adjust geographic transformation for pixel size
        new_tr = (transf[0], transf[1] * ksize, 0, transf[3], 0, transf[5] * ksize)
    
    iter_rows = min((img1_wins.shape[0], img2_wins.shape[0]))
    iter_cols = min((img1_wins.shape[1], img2_wins.shape[1]))

    # Only process tiles that are not completely black
    # flatten the tile arrays
    img1_wins = img1_wins.reshape(-1, ksize, ksize)
    img2_wins = img2_wins.reshape(-1, ksize, ksize)
    good_locs_img1 = np.where(
        np.any(img1_wins.reshape(-1, ksize ** 2) != 0, axis=1)
        )[0]
    good_locs_img2 = np.where(
        np.any(img2_wins.reshape(-1, ksize ** 2) != 0, axis=1)
        )[0]
    common_good_locs = set(good_locs_img1).intersection(set(good_locs_img2))

    # Get displacement map
    disp_map = np.zeros((iter_rows, iter_cols), dtype=np.float16).flatten()
    for loc in common_good_locs:
        disp_map[loc] = np.round(kernel_disp(img1_wins[loc], img2_wins[loc]), 1)
    
    # Bring back to correct shape
    disp_map = disp_map.reshape((iter_rows, iter_cols))

    return disp_map, new_tr

def estimate_disp(disp_map: np.ndarray) -> np.ndarray:
    """
    Method to estimate subpixel displacement.

    :param disp_map: Displacement map obtained from the phase_correlation method
    :return: Displacement value
    """

    if disp_map.ndim == 3:
        nrow, ncol, _ = disp_map.shape  # Get # of rows in correlation surface
    elif disp_map.ndim == 2:
        nrow, ncol = disp_map.shape  # Get # of rows in correlation surface
    peak_y, peak_x = np.unravel_index(
        np.argmax(disp_map, axis=None), disp_map.shape
    )

    # Get displacements adjacent to peak
    x_bef = (peak_x - 1 >= 0) * (peak_x - 1) + (peak_x - 1 < 0) * peak_x
    x_aft = (peak_x + 1 >= ncol - 1) * peak_x + (peak_x + 1 < ncol - 1) * (
        peak_x + 1
    )
    y_bef = (peak_y - 1 >= 0) * (peak_y - 1) + (peak_y - 1 < 0) * peak_y
    y_aft = (peak_y + 1 >= nrow - 1) * peak_y + (peak_y + 1 < nrow - 1) * (
        peak_y + 1
    )

    # Estimate subpixel displacement in x-direction
    dx_num = np.log(disp_map[peak_y, x_aft]) - np.log(disp_map[peak_y, x_bef])
    dx_denom = 2 * (
        2 * np.log(disp_map[peak_y, peak_x])
        - np.log(disp_map[peak_y, x_aft])
        - np.log(disp_map[peak_y, x_bef])
    )
    dx = dx_num / dx_denom  # subpixel motion in x direction (East/West)
    if math.isnan(dx):
        dx = 0.0

    # Estimate subpixel displacement in y-direction
    dy_num = np.log(disp_map[y_aft, peak_x]) - np.log(disp_map[y_bef, peak_x])
    dy_denom = 2 * (
        2 * np.log(disp_map[peak_y, peak_x])
        - np.log(disp_map[y_aft, peak_x])
        - np.log(disp_map[y_bef, peak_x])
    )
    dy = dy_num / dy_denom  # subpixel motion in y direction (North/South)

    if math.isnan(dy):
        dy = 0.0
    if np.abs(peak_x) > disp_map.shape[1] / 2:
        peak_x = (
            peak_x - disp_map.shape[1]
        )  # convert x offsets > ws/2 to negative offsets
    if np.abs(peak_y) > disp_map.shape[0] / 2:
        peak_y = (
            peak_y - disp_map.shape[0]
        )  # convert y offsets > ws/2 to negative offsets

    disx = peak_x + dx
    disy = peak_y + dy
    dis = np.sqrt(np.power(disx, 2) + np.power(disy, 2))
    
    return dis
