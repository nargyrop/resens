import logging
from typing import Callable, Sequence, Tuple, Type, Union

import numpy as np
from numpy import int8, short, single, uint8, ushort

logger = logging.getLogger(__name__)

__all__ = ["find_dtype", "swf", "get_sliding_win", "get_tiles"]


def find_dtype(
    in_arr: np.ndarray,
) -> Tuple[str, Type[Union[int8, uint8, short, ushort, single]]]:
    """
    Method to retrieve an array's data type.

    :param in_arr: Image Array
    :return: Array type, numpy dtype
    """

    # Initialize a dictionary to store the numpy attribute for each dtype
    dtype_dict = {
        "int8": np.int8,
        "int16": np.int16,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "float32": np.float32,
    }

    # Get minimum and maximum values in the array and the value of one random element
    min_val = np.min(in_arr)
    max_val = np.max(abs(in_arr))

    if np.array_equal(in_arr, in_arr.astype(np.int_)):
        if max_val < 256:
            if min_val < 0:
                arrtype = "int8"
                npdtype = dtype_dict[arrtype]
            else:
                arrtype = "uint8"
                npdtype = dtype_dict[arrtype]
        elif 256 <= max_val <= 65535:
            if min_val < 0:
                arrtype = "int16"
                npdtype = dtype_dict[arrtype]
            else:
                arrtype = "uint16"
                npdtype = dtype_dict[arrtype]
        else:
            arrtype = "float32"
            npdtype = dtype_dict[arrtype]
    else:
        arrtype = "float32"
        npdtype = dtype_dict[arrtype]

    return arrtype, npdtype


def swf(
    in_arr: np.ndarray, ksize: int = None, filter_op: Union[Callable, str] = "mean"
) -> np.ndarray:
    """
    Method to apply a pixel-by-pixel median or mean filter using the side
    window technique.

    :param in_arr: Input array (2D or 3D numpy array)
    :param ksize: Window size (odd number). If even, will be adjusted to the
                  next odd number.
    :param filter_op: Filter operation to be used. Can be "median", "mean",
                      or a callable function. Defaults to "mean".
    :return: Filtered array with the same shape and dtype as input
    """
    from numpy.lib.stride_tricks import as_strided

    # Available filter operations
    if isinstance(filter_op, str):
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
    up_down_meds = np.apply_over_axes(filter_op, up_down, pr_axes).astype(up_down.dtype)
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
    inds = np.argmin(subtr, axis=0)  # Get indices where the subtr is minimum along the 0
    # axis
    subtr = None

    # Get the output pixel values
    if not bands:
        filt = np.take_along_axis(stacked, np.expand_dims(inds, axis=0), axis=0).reshape(
            sy, sx
        )
    else:
        filt = np.take_along_axis(stacked, np.expand_dims(inds, axis=0), axis=0).reshape(
            sy, sx, bands
        )

    stacked = None

    return filt


def get_sliding_win(
    array: np.ndarray,
    ksize: int,
    step_x: int = 1,
    step_y: int = 1,
    pad: bool = True,
) -> np.ndarray:
    """
    Efficient method that returns sliced arrays for sliding windows.

    :param array: Image array
    :param ksize: Odd integer window size
    :param step_x: Step or stride size in the x-direction (def=1)
    :param step_y: Step or stride size in the y-direction (def=1)
    :param pad: Flag to enable image padding equal to the radius of ksize
    :return: 4D array matching the input array's size. Each element is an array
    matching the window size+bands
    """

    from numpy.lib.stride_tricks import as_strided

    # Get window radius, padded array and strides
    if pad:
        radius = ksize // 2
        pad_widths = [
            [radius, radius],  # 0-axis padding
            [radius, radius],  # 1-axis padding
        ]
        if array.ndim == 3:
            pad_widths += [[0, 0]]  # 2-axis padding (no padding)
        array = np.pad(array, pad_widths, "reflect")
    else:
        radius = 0

    if array.ndim == 2:
        sy, sx = array.shape
        nbands = False
    elif array.ndim == 3:
        sy, sx, nbands = array.shape
    else:
        raise ValueError(f"Incorrect array shape {array.shape}")

    # Calculate output shape
    y_size = int(np.floor((sy + 2 * radius - ksize) / step_y) + 1)
    x_size = int(np.floor((sx + 2 * radius - ksize) / step_x) + 1)

    if not nbands:
        strides = (
            array.strides[0] * step_y,
            array.strides[1] * step_x,
        ) + array.strides
        out_shape = (
            y_size,
            x_size,
            ksize,
            ksize,
        )
    else:
        strides = (
            array.strides[0] * step_y,
            array.strides[1] * step_x,
            array.strides[2],
        ) + array.strides
        out_shape = (y_size, x_size, 1, ksize, ksize, nbands)

    # Slice the padded array using strides
    sliced_array = np.squeeze(as_strided(array, shape=out_shape, strides=strides))

    return sliced_array


def get_tiles(
    array: np.ndarray,
    ksize: Union[int, Sequence[int]] = None,
    nblocks: int = None,
    pad: bool = True,
) -> np.ndarray:
    """
    Efficient method that returns sliced arrays for sliding windows.

    :param array: Image array
    :param ksize: Integer window size or List/Tuple of window sizes in x and y
    directions
    :param nblocks: Integer number of tiles in which to divide the array or
    List/Tuple of number ot tiles in x and y directions
    :param pad: Flag to enable image padding equal to the radius of ksize
    :return: 4D array matching the input array's size. Each element is an array
    matching the window size+bands
    """

    from numpy.lib.stride_tricks import as_strided

    if ksize:
        if isinstance(ksize, (list, tuple)):
            ksize_x = ksize[0]
            ksize_y = ksize[1]
            step_x = ksize[0]
            step_y = ksize[1]
        else:
            ksize_x = ksize
            ksize_y = ksize
            step_x = ksize
            step_y = ksize
    elif nblocks:
        if isinstance(nblocks, (list, tuple)):
            ksize_x = array.shape[1] // nblocks[0]
            ksize_y = array.shape[0] // nblocks[1]
            step_x = ksize_x
            step_y = ksize_y
        else:
            ksize_x = array.shape[1] // nblocks
            ksize_y = array.shape[0] // nblocks
            step_x = ksize_x
            step_y = ksize_y
    else:
        raise RuntimeError("One of `ksize`, `nblocks` needs to be passed!")

    # Get window radius, padded array and strides
    if pad:
        pad_widths = [
            [0, ksize - array.shape[0] % ksize],  # 0-axis padding
            [0, ksize - array.shape[1] % ksize],  # 1-axis padding
        ]
        if array.ndim == 3:
            pad_widths += [[0, 0]]  # 2-axis padding (no padding)
        array = np.pad(array, pad_widths, "reflect")

    if array.ndim == 2:
        sy, sx = array.shape
        nbands = False
    elif array.ndim == 3:
        sy, sx, nbands = array.shape
    else:
        raise ValueError(f"Incorrect array shape {array.shape}")

    # Calculate output shape
    if not nbands:
        strides = (
            array.strides[0] * step_y,
            array.strides[1] * step_x,
        ) + array.strides
        out_shape = (sy // step_y, sx // step_x, ksize_y, ksize_x)
    else:
        strides = (
            array.strides[0] * step_y,
            array.strides[1] * step_x,
            array.strides[2],
        ) + array.strides
        out_shape = (sy // step_y, sx // step_x, 1, ksize_y, ksize_x, nbands)

    # Slice the padded array using strides
    sliced_array = np.squeeze(as_strided(array, shape=out_shape, strides=strides))

    return sliced_array
