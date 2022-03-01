import math
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type, Union

import cv2
import numpy as np
from numpy import int8, short, single, uint8, ushort

try:
    import gdal
    import osr
except ImportError:
    from osgeo import gdal
    from osgeo import osr

from rasteroptions import CO_COMPRESS, CO_NOCOMPRESS, GDAL_DTYPES


class Analysis:
    def __init__(self):
        pass

    @staticmethod
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

    def kernel_disp(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Method to get the mean displacement within a tile.

        :param img1: Image1 tile
        :param img2: Image2 tile
        :return: Mean subpixel displacement
        """

        # Phase correlation
        fft_img1 = np.fft.fft2(img1)  # FFT
        fft_img2 = np.fft.fft2(img2)
        conj_img2 = np.ma.conjugate(fft_img2)  # Complex conjugate
        R_12 = fft_img1 * conj_img2  # Cross-power spectrum
        R_12 /= np.absolute(R_12)
        disp_map_12 = np.fft.ifft2(R_12).real  # Normalized cross-correlation

        # Estimate mean displacement in pixels for each band pair
        mean_disp = np.mean(self.estimate_disp(disp_map_12))

        return mean_disp

    def phase_correlation(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        ksize: int = 3,
        eq_histogram: bool = None,
        transf: Union[Tuple, List] = None,
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Wrapper method to estimate subpixel displacement between 2 grayscale
        images.

        :param img1: Grayscale image array
        :param img2: Grayscale image array
        :param ksize: Kernel size
        :param eq_histogram: Enable histogram equalization
        :param transf: Geo-transformation tuple
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

        # Adjust geographic transformation for pixel size
        new_tr = (transf[0], transf[1] * ksize, 0, transf[3], 0, transf[5] * ksize)

        # Get tiles for each band
        img1_wins = Processing.get_tiles(in_arr=img1, ksize=ksize)
        img2_wins = Processing.get_tiles(in_arr=img2, ksize=ksize)
        iter_rows = min((img1_wins.shape[0], img2_wins.shape[0]))
        iter_cols = min((img1_wins.shape[1], img2_wins.shape[1]))

        # Get displacement map
        disp_map = np.array(
            [
                [
                    self.kernel_disp(img1_wins[i][j], img2_wins[i][j])
                    for j in range(iter_cols)
                ]
                for i in range(iter_rows)
            ]
        )

        return disp_map, new_tr

    @staticmethod
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


class IO:
    def __init__(self):
        pass

    @staticmethod
    def load_image(
        img_path: Union[Path, str],
    ) -> Iterable:
        """
        Method to load an array from a raster file and retrieve the geo-
        transformation, projection and EPSG of the CRS.

        :param img_path: Path to image
        :return: tuple containing: Array, geo-transformation, projection, epsg code
        """

        if isinstance(img_path, Path):
            img_path = img_path.as_posix()
        obj = gdal.Open(img_path)
        array = obj.ReadAsArray()
        if array.ndim == 3:
            array = np.einsum("ijk->jki", array)
        array = array.astype(Utils.find_dtype(array)[1])

        transf = obj.GetGeoTransform()
        proj = obj.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        epsg = srs.GetAttrValue("AUTHORITY", 1)

        # Check that the pixel sizes are of the correct sign
        xo, psx, skx, yo, sky, psy = list(transf)
        if psy > 0:
            psy *= -1
        transf = (xo, psx, skx, yo, sky, psy)

        return array, transf, proj, epsg

    def load_from_zip(
        self,
        zipf_path: Union[Path, str],
        req_files: Union[list, tuple],
        extension: str,
    ) -> Union[Dict, None]:
        """
        Method that loads all the required bands in arrays and saves them to a
        dictionary.

        :param zipf_path: Path to zip file
        :param req_files: List of strings included in the file names (e.g. band numbers)
        :param extension: Extension of the target image
        :return: Dictionary containing the array, geo-transformation tuple, projection
        and EPSG code of each image.
        List containing the dictionary keys
        """

        # Check if req_files is actually a list
        if not isinstance(req_files, (list, tuple)):
            req_files = [req_files]

        # Check if the zip file path is correct
        if isinstance(zipf_path, str):
            zipf_path = Path(zipf_path)
        if not zipf_path.exists():
            raise FileNotFoundError(f"Zip file {zipf_path} does not exist!")

        # Initialize gdal zip file handler
        ziphandler = "/vsizip/"

        # Read zip file
        try:
            archive = zipfile.ZipFile(zipf_path, "r")
        except zipfile.BadZipfile:
            return None
        else:
            # Get the zip structure for the required bands
            img_ls = [
                f
                for f in archive.namelist()
                if f.endswith(extension) and any(x in f for x in req_files)
            ]

            # Create dictionaries to store the data
            band_dict = {}

            for img in img_ls:
                try:
                    # Find which of the req files fits the current, create the dict key
                    # and store it in the keys' list
                    (key_in,) = [key for key in req_files if key in img]

                    # Load image, get metadata and store to dictionary
                    array, transf, proj, epsg = self.load_image(
                        ziphandler + zipf_path.joinpath(img).as_posix()
                    )
                    band_dict[key_in] = [array, transf, proj, epsg]
                except AttributeError:
                    raise AttributeError(f"Error loading {img}")

            return band_dict

    @staticmethod
    def write_image(
        out_arr: np.ndarray,
        output_img: Union[Path, str],
        transformation: Tuple,
        projection: str,
        nodata: Union[int, float] = None,
        compression: bool = True,
        datatype: str = None,
    ):
        """
        Method that writes an array to a georeferenced GeoTIFF file.

        :param out_arr: Array containing the mask.
        :param output_img: Output image path.
        :param transformation: Geometric Transformation (Format: (Xo, pixel size in X
        direction, X-axis skew, Yo, Y-axis skew, pixel size in Y direction)).
        :param projection: Projection string.
        :param nodata: NoData value.
        :param compression: True to enable compression (default), False to disable.
        :param datatype: Array datatype. Set to None to have the script automatically
        detect the datatype or select
        between uint8, uint16, int8, int16, float32.
        :return:
        """

        # Dtype Dictionary to be used for output

        # Check that the pixel sizes are of the correct sign
        xo, psx, skx, yo, sky, psy = list(transformation)
        if psy > 0:
            psy *= -1
        transformation = (xo, psx, skx, yo, sky, psy)

        # Get array type
        if datatype is None:
            datatype, _ = Utils.find_dtype(out_arr)

        out_arr = out_arr.astype(datatype)
        gdal_datatype = GDAL_DTYPES[datatype]

        try:
            # Determine the shape of the array and the number of bands
            if out_arr.shape[0] > out_arr.shape[2]:
                row_ind = 0
                col_ind = 1
                nband = out_arr.shape[2]
            else:
                row_ind = 1
                col_ind = 2
                nband = out_arr.shape[0]

        except IndexError:
            row_ind = 0
            col_ind = 1
            nband = 1

        # Construct output image path
        if isinstance(output_img, str):
            output_img = Path(output_img)
        if output_img.suffix == "":
            output_img = output_img.with_suffix(".tif")

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            output_img.as_posix(),
            out_arr.shape[col_ind],
            out_arr.shape[row_ind],
            nband,
            gdal_datatype,
            options=CO_COMPRESS if compression else CO_NOCOMPRESS,
        )
        dataset.SetGeoTransform(transformation)
        dataset.SetProjection(projection)

        for i in range(nband):
            if not nband == 1:
                out_band = dataset.GetRasterBand(i + 1)
                if nodata:
                    out_band.SetNoDataValue(nodata)
                out_band.WriteArray(out_arr[..., i])
            else:
                out_band = dataset.GetRasterBand(i + 1)
                if nodata:
                    out_band.SetNoDataValue(nodata)
                out_band.WriteArray(out_arr)

        dataset = None


class Processing:
    def __init__(self):
        pass

    @staticmethod
    def resample_array(
        in_arr: np.ndarray,
        out_shape: Union[Tuple, List] = None,
        in_pix: Union[float, int] = None,
        out_pix: Union[float, int] = None,
    ) -> np.ndarray:
        """
        Method that resamples arrays using the shape or the pixel size.

        :param in_arr: 2D/3D Image array
        :param out_shape: Tuple of output array dimension (e.g. (nrows, ncols))
        :param in_pix: Input pixel size. Provide instead of out_shape. For non-square
        pixels, provide a tuple (psy, psx)
        :param out_pix: Output pixel size. Provide along with in_pix instead of out_shape.
        For non-square pixels, provide a tuple (psy, psx)
        :return: Resampled array, Adjusted Geo-transformation
        """

        # Make sure in_arr is of a supported dtype
        try:
            assert Utils.find_dtype(in_arr) in ("uint8", "uint16", "float32")
        except AssertionError:
            in_arr = in_arr.astype(Utils.find_dtype(in_arr)[1])

        # Resize array
        if out_shape:
            if not in_arr.shape == out_shape:
                resampled = cv2.resize(in_arr, out_shape)
            else:
                resampled = in_arr
        elif in_pix and out_pix:
            if not isinstance(in_pix, (list, tuple)) and not isinstance(
                out_pix, (list, tuple)
            ):
                # Square pixels
                scalr = in_pix / out_pix
                resampled = cv2.resize(in_arr, None, fx=scalr, fy=scalr)
            else:
                # Rectangular pixels
                scalrx = in_pix[1] / out_pix[1]
                scalry = in_pix[0] / out_pix[0]
                resampled = cv2.resize(in_arr, None, fx=scalrx, fy=scalry)
        else:
            resampled = None

        return resampled

    @staticmethod
    def convert8bit(nbit_image: np.ndarray) -> np.ndarray:
        """
        Method to convert any n-Bit image to 8-bit with contrast enhancement
        (histogram truncation).

        :param nbit_image: 2D or 3D array
        :return: 8-bit image
        """

        def _make_8bit(arr: np.ndarray) -> np.ndarray:
            # Get image statistics
            av_val = np.mean(arr)
            std_val = np.std(arr)
            min_val = av_val - 1.96 * std_val
            min_val = (
                min_val if min_val >= 0 else 0
            )  # make sure min value is not negative
            max_val = av_val + 1.96 * std_val

            # Truncate the array - Contrast Enhancement
            arr[arr > max_val] = max_val
            arr[arr < min_val] = min_val

            # Convert to 8bits
            arr = np.divide(arr - min_val, max_val - min_val) * 255

            return arr

        # Iterate over each array and get min and max corresponding to a 5-95% data
        # truncation
        # Also, resample if needed to the largest size
        bit8_img = np.dstack(nbit_image).astype(np.float32)
        nbit_image = None

        if bit8_img.ndim == 3:
            for i in range(bit8_img.shape[2]):
                bit8_img[..., i] = _make_8bit(bit8_img[..., i])
        else:
            # Convert to 8bits
            bit8_img = _make_8bit(bit8_img)

        return bit8_img

    @staticmethod
    def multiband2grayscale(img: np.ndarray) -> np.ndarray:
        """
        Function to convert point coordinates to geojson format.

        :param img: Multichannel image array.
        :return: Grayscale image
        """

        # If the image is not composed of multiple channels, it is already grayscale
        try:
            assert img.ndim == 3
        except AssertionError:
            return img

        img_sum = np.sum(img, axis=2)  # Sum of all channel pixels

        # Compute weight for each channel
        img_weights_arr = np.dstack(
            [np.divide(img[..., i], img_sum) for i in range(img.shape[2])]
        )
        channel_weights = np.nanmean(img_weights_arr, axis=(0, 1))
        while channel_weights.sum() > 1.0:
            channel_weights -= (
                channel_weights * 0.01
            )  # Make sure the weights sum to 1.0

        # Get grayscale image
        gray = np.sum(img * channel_weights, axis=2)

        return gray

    @staticmethod
    def get_sliding_win(
        in_arr: np.ndarray,
        ksize: int,
        step_x: int = 1,
        step_y: int = 1,
        pad: bool = True,
    ) -> np.ndarray:
        """
        Efficient method that returns sliced arrays for sliding windows.

        :param in_arr: 2D or 3D array
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
            ksize += 1 - ksize % 2  # Make sure window size is an odd number
            radius = ksize // 2
            pad_widths = [
                [radius, radius],  # 0-axis padding
                [radius, radius],  # 1-axis padding
            ]
            if in_arr.ndim == 3:
                pad_widths += [[0, 0]]  # 2-axis padding (no padding)
            in_arr = np.pad(in_arr, pad_widths, "reflect")
        else:
            radius = 0

        if in_arr.ndim == 2:
            sy, sx = in_arr.shape
            nbands = False
        elif in_arr.ndim == 3:
            sy, sx, nbands = in_arr.shape
        else:
            raise ValueError(f"Incorrect array shape {in_arr.shape}")

        # Calculate output shape
        if not nbands:
            strides = (
                in_arr.strides[0] * step_y,
                in_arr.strides[1] * step_x,
            ) + in_arr.strides
            out_shape = (
                (sy - ksize + 2 * radius + step_y) // step_y,
                (sx - ksize + 2 * radius + step_x) // step_x,
                ksize,
                ksize,
            )
        else:
            strides = (
                in_arr.strides[0] * step_y,
                in_arr.strides[1] * step_x,
                in_arr.strides[2],
            ) + in_arr.strides
            out_shape = (
                (sy - ksize + 2 * radius + step_y) // step_y,
                (sx - ksize + 2 * radius + step_x) // step_x,
                1,
                ksize,
                ksize,
                nbands,
            )

        # Slice the padded array using strides
        sliced_array = as_strided(in_arr, shape=out_shape, strides=strides)

        return sliced_array

    @staticmethod
    def get_tiles(
        in_arr: np.ndarray,
        ksize: Union[int, List[int], Tuple[int]] = None,
        nblocks: int = None,
    ) -> np.ndarray:
        """
        Efficient method that returns sliced arrays for sliding windows.

        :param in_arr: 2D or 3D array
        :param ksize: Integer window size or List/Tuple of window sizes in x and y
        directions
        :param nblocks: Integer number of tiles in which to divide the array or
        List/Tuple of number ot tiles in x and y directions
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
                ksize_x = in_arr.shape[1] // nblocks[0]
                ksize_y = in_arr.shape[0] // nblocks[1]
                step_x = ksize_x
                step_y = ksize_y
            else:
                ksize_x = in_arr.shape[1] // nblocks
                ksize_y = in_arr.shape[0] // nblocks
                step_x = ksize_x
                step_y = ksize_y

        if in_arr.ndim == 2:
            sy, sx = in_arr.shape
            nbands = False
        elif in_arr.ndim == 3:
            sy, sx, nbands = in_arr.shape
        else:
            raise ValueError(f"Incorrect array shape {in_arr.shape}")

        # Calculate output shape
        if not nbands:
            strides = (
                in_arr.strides[0] * step_y,
                in_arr.strides[1] * step_x,
            ) + in_arr.strides
            out_shape = (sy // step_y, sx // step_x, ksize_y, ksize_x)
        else:
            strides = (
                in_arr.strides[0] * step_y,
                in_arr.strides[1] * step_x,
                in_arr.strides[2],
            ) + in_arr.strides
            out_shape = (sy // step_y, sx // step_x, 1, ksize_y, ksize_x, nbands)

        # Slice the padded array using strides
        sliced_array = as_strided(in_arr, shape=out_shape, strides=strides)

        return sliced_array


class Utils:
    def __init__(self):
        pass

    @staticmethod
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
        max_val = np.max(in_arr)

        if np.array_equal(in_arr, in_arr.astype(np.int)):
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

    @staticmethod
    def shapefile_masking(
        polygon_shp: Union[Path, str],
        shape: Union[tuple, list],
        transformation: tuple,
        projection: str,
        mask_outpath: Union[Path, str] = None,
        burn_value: Union[int, float] = 1,
        dilation: bool = False,
        dilation_iters: int = None,
        compression: bool = True,
    ) -> np.ndarray:
        """
        Rasterize a polygons shapefile and create a raster mask.

        :param polygon_shp: Absolute path to land polygon shapefile
        :param mask_outpath: Absolute path (including filename) to output raster mask
        :param shape: Input array's size including channels (e.g. (640, 480, 3))
        :param transformation: Geographic transformation tuple
        :param projection: CRS Projection
        :param burn_value: Value to burn to output raster
        :param dilation: Flag to enable dilating the land mask
        :param dilation_iters: Number of dilation iterations
        :param compression: True to enable compression (default), False to disable.
        :return: Binary mask array
        """

        # Set up the output filename in a way that it won't be needed to create a
        # mask for arrays with the same extents
        if isinstance(polygon_shp, str):
            polygon_shp = Path(polygon_shp)
        if isinstance(mask_outpath, str):
            mask_outpath = Path(mask_outpath)
        if mask_outpath.suffix != ".tif":
            mask_outpath = mask_outpath.joinpath(f"land_mask_{str(uuid.uuid4())}.tif")

        if not mask_outpath.exists():
            # Write empty raster
            mask_arr = np.zeros(shape[:2], dtype=np.int8)
            IO.write_image(
                mask_arr,
                mask_outpath.as_posix(),
                transformation,
                projection,
                compression,
            )

            # Rasterization
            _ = subprocess.run(
                [
                    "gdal_rasterize",
                    "-burn",
                    str(burn_value),
                    polygon_shp.as_posix(),
                    mask_outpath.as_posix(),
                ],
                stdout=subprocess.PIPE,
            ).stdout.decode("utf-8")

        # Load the mask array
        mask_arr, _, _, _ = IO.load_image(mask_outpath.as_posix())

        # Dilate the mask
        if dilation:
            if dilation_iters:
                mask_arr = cv2.dilate(
                    mask_arr,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                    iterations=int(dilation_iters),
                )
            else:
                mask_arr = cv2.dilate(
                    mask_arr,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                    iterations=1,
                )

        return mask_arr
