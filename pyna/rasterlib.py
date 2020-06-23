import os
import cv2
import osr
import gdal
import zipfile
import numpy as np


class Raster:

    def __init__(self):
        pass

    @staticmethod
    def remove_var(var):
        """
        Method to remove unwanted objects from memory

        :type var: Object or List/Tuple of objects
        :param var: Object or List/Tuple of objects to be removed from memory

        :return: Nothing
        """
        if isinstance(var, (list, tuple)):
            for i in var:
                i = None
        else:
            var = None

    @staticmethod
    def __dtype(in_arr):
        """
        Method to retrieve an array's data type.

        :type in_arr: Array
        :param in_arr: Image Array

        :return: Array type
        """

        # Initialize a dictionary to store the numpy attribute for each dtype
        dtype_dict = {'int8': np.int8, 'int16': np.int16, 'uint8': np.uint8, 'uint16': np.uint16, 'float32': np.float32}

        # Get minimum and maximum values in the array and the value of one random element
        min_val = in_arr.min()
        max_val = in_arr.max()

        if np.array_equal(in_arr, in_arr.astype(np.int)):
            if max_val < 256:
                if min_val < 0:
                    arrtype = 'int8'
                    npdtype = dtype_dict[arrtype]
                else:
                    arrtype = 'uint8'
                    npdtype = dtype_dict[arrtype]
            elif 256 <= max_val <= 65535:
                if min_val < 0:
                    arrtype = 'int16'
                    npdtype = dtype_dict[arrtype]
                else:
                    arrtype = 'uint16'
                    npdtype = dtype_dict[arrtype]
            else:
                arrtype = 'float32'
                npdtype = dtype_dict[arrtype]
        else:
            arrtype = 'float32'
            npdtype = dtype_dict[arrtype]

        return arrtype, npdtype

    """I/O"""

    def load_image(self, img_path):
        """
        Method to load an array from a raster file and retrieve the geo-transformation, projection and EPSG of the CRS

        :type img_path: String
        :param img_path: Path to image

        :return: Array, geo-transformation, projection, epsg code
        """

        obj = gdal.Open(img_path)
        array = obj.ReadAsArray()
        if len(array.shape) == 3:
            array = np.einsum('ijk->jki', array)
        array = array.astype(self.__dtype(array)[1])

        transf = obj.GetGeoTransform()
        proj = obj.GetProjection()
        srs = osr.SpatialReference(wkt=proj)
        epsg = srs.GetAttrValue('AUTHORITY', 1)

        return array, transf, proj, epsg

    def load_from_zip(self, zipf_path, req_files, extension):
        """
        Method that loads all the required bands in arrays and saves them to a dictionary.

        :type zipf_path: String
        :param zipf_path: Path to zip file

        :type req_files: String or List/Tuple of strings
        :param req_files: List of strings included in the file names (e.g. band numbers)

        :type extension: String
        :param extension: Extension of the target image

        :return: Dictionary containing the array, geo-transformation tuple, projection and EPSG code of each image.
        List containing the dictionary keys
        """

        # Initialize gdal zip file handler
        ziphandler = '/vsizip/' + zipf_path

        # Read zip file
        try:
            archive = zipfile.ZipFile(zipf_path, 'r')
        except zipfile.BadZipfile:
            pass

        # Check if req_files is actually a list
        if not isinstance(req_files, (list, tuple)):
            req_files = [req_files]

        # Get the zip structure for the required bands
        img_ls = [f for f in archive.namelist() if f.endswith(extension) and any(x in f for x in req_files)]

        # Create dictionaries to store the data
        band_dict = {}

        for img in img_ls:
            try:
                # Find which of the req files fits the current, create the dict key and store it in the keys' list
                key_in = os.path.split(img)[1].split('.')[0]

                # Load image, get metadata and store to dictionary
                array, transf, proj, epsg = self.load_image(os.path.join(ziphandler, img))
                band_dict[key_in] = [array, transf, proj, epsg]
            except AttributeError:
                pass

        return band_dict

    def write_image(self, out_arr, output_img, transf, prj, compression=True):
        """
        Method that writes an array to a georeferenced GeoTIFF file.

        :type out_arr: 2D binary image array
        :param out_arr: Array containing the mask

        :type output_img: String
        :param output_img: Output image path

        :type transf: Tuple of floats :param transf: Geometric Transformation (Format: (Xo, pixel size in X
        direction, X-axis skew, Yo, Y-axis skew, pixel size in Y direction))

        :type prj: String
        :param prj: Projection

        :type compression: Boolean
        :param compression: True to enable compression (default), False to disable

        :return: Nothing
        """

        # Dtype Dictionary to be used for output
        gdal_dtype = {'uint8': gdal.GDT_Byte,
                      'uint16': gdal.GDT_UInt16,
                      'int8': gdal.GDT_Byte,
                      'int16': gdal.GDT_Int16,
                      'float32': gdal.GDT_Float32}

        # Get array type
        out_arr = out_arr.astype(self.__dtype(out_arr)[0])

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

        # Set output options for compression
        if compression:
            opt_ls = ['TILED=YES', 'COMPRESS=DEFLATE', 'NUM_THREADS=ALL_CPUS']
        else:
            opt_ls = []

        if not output_img.endswith('.tif'):
            output_img = f"{output_img}.tif"
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_img, out_arr.shape[col_ind], out_arr.shape[row_ind], nband,
                                gdal_dtype[self.__dtype(out_arr)[0]], options=opt_ls)
        dataset.SetGeoTransform(transf)
        dataset.SetProjection(prj)

        for i in range(nband):
            if not nband == 1:
                dataset.GetRasterBand(i + 1).WriteArray(out_arr[:, :, i])
            else:
                dataset.GetRasterBand(i + 1).WriteArray(out_arr[:, :])

        dataset = None

    """Masking"""

    def land_masking(self, shape, transf, prj, land_shp, land_mask_path, dilation=False, dilation_iters=None):
        """
        Method to rasterize the land polygons shapefile and create a raster land mask.

        :type shape: Tuple
        :param shape: Input array's size including channels (e.g. (640, 480, 3))

        :type transf: Tuple
        :param transf: Geographic transformation tuple

        :type prj: String
        :param prj: CRS Projection

        :type land_shp: String
        :param land_shp: Absolute path to land polygon shapefile

        :type land_mask_path: String
        :param land_mask_path: Absolute path to output raster land mask directory

        :return: Binary land mask array (0-1)
        """

        # Set up the output filename in a way that it won't be needed to create a land mask for arrays with the same
        # extents
        land_mask_fname = 'land_mask_{}_{}_{}_{}_{}_{}'.format(round(transf[0], 3), round(transf[3], 3), transf[1],
                                                               abs(transf[-1]), shape[1], shape[0])

        if not os.path.exists(os.path.join(land_mask_path, f"{land_mask_fname}.tif")):

            # Create an array
            if len(shape) > 2:
                land_mask = np.zeros(shape[:2])
            else:
                land_mask = np.zeros(shape)

            # First create an empty file matching the input array's dimensions
            self.write_image(land_mask, os.path.join(land_mask_path,  land_mask_fname), transf, prj, True)

            # Then burn the shapefile on the raster file
            os.system(f'gdal_rasterize -burn 1 {land_shp} {os.path.join(land_mask_path, f"{land_mask_fname}.tif")}')

        # Load the land mask array
        land_mask_arr, _, _, _ = self.load_image(os.path.join(land_mask_path, f"{land_mask_fname}.tif"))

        if dilation:
            if dilation_iters:
                land_mask_arr = cv2.dilate(land_mask_arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                                           iterations=int(dilation_iters))
            else:
                land_mask_arr = cv2.dilate(land_mask_arr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                                           iterations=1)
        else:
            pass

        return land_mask_arr

    """Fast Sliding Windows and Filters"""

    @staticmethod
    def get_sliding_win(in_arr, ksize, step_x=1, step_y=1):

        """
        Efficient method that returns sliced arrays for sliding windows.

        :type in_arr: Array
        :param in_arr: 2D or 3D array

        :type ksize: Int
        :param ksize: Odd integer window size

        :type step_x: Int
        :param step_x: Step or stride size in the x-direction (def=1)

        :type step_y: Int
        :param step_y: Step or stride size in the y-direction (def=1)

        :return: 4D array matching the input array's size. Each element is an array matching the window size+bands
        """

        from numpy.lib.stride_tricks import as_strided

        # Get window radius, padded array and strides
        ksize += 1 - ksize % 2  # Make sure window size is an odd number
        radius = ksize // 2
        padded = np.pad(in_arr, radius, 'reflect')
        try:
            assert len(in_arr.shape) == 2
            sy, sx = in_arr.shape
            nbands = False
        except AssertionError:
            sy, sx, nbands = in_arr.shape

        # Calculate output shape
        if not nbands:
            strides = (padded.strides[0] * step_y, padded.strides[1] * step_x) + padded.strides
            out_shape = ((sy - ksize + 2 * radius + step_y) // step_y,
                         (sx - ksize + 2 * radius + step_x) // step_x,
                         ksize, ksize)
        else:
            strides = (padded.strides[0] * step_y, padded.strides[1] * step_x, padded.strides[2]) + padded.strides
            out_shape = ((sy - ksize + 2 * radius + step_y) // step_y,
                         (sx - ksize + 2 * radius + step_x) // step_x, 1,
                         ksize, ksize, nbands)

        # Slice the padded array using strides
        sliced_array = as_strided(padded, shape=out_shape, strides=strides)

        return sliced_array

    @staticmethod
    def get_tiles(in_arr, ksize=None, nblocks=None):

        """
        Efficient method that returns sliced arrays for sliding windows.

        :type in_arr: Array
        :param in_arr: 2D or 3D array

        :type ksize: Int or List/Tuple
        :param ksize: Integer window size or List/Tuple of window sizes in x and y directions

        :type nblocks: Int
        :param nblocks: Integer number of tiles in which to divide the array or List/Tuple of number ot tiles in x and y directions

        :return: 4D array matching the input array's size. Each element is an array matching the window size+bands
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
        try:
            assert len(in_arr.shape) == 2
            sy, sx = in_arr.shape
            nbands = False
        except AssertionError:
            sy, sx, nbands = in_arr.shape

        # Calculate output shape
        if not nbands:
            strides = (in_arr.strides[0] * step_y, in_arr.strides[1] * step_x) + in_arr.strides
            out_shape = (sy // step_y,
                         sx // step_x,
                         ksize_y, ksize_x)
        else:
            strides = (in_arr.strides[0] * step_y, in_arr.strides[1] * step_x, in_arr.strides[2]) + in_arr.strides
            out_shape = (sy // step_y,
                         sx // step_x, 1,
                         ksize_y, ksize_x, nbands)

        # Slice the padded array using strides
        sliced_array = as_strided(in_arr, shape=out_shape, strides=strides)

        return sliced_array

    def swf(self, in_arr, ksize=None, filter_op='mean'):
        """
        Method to apply a pixel-by-pixel median or mean filter using the side window technique.

        :type in_arr: Array
        :param in_arr: 2D/3D array

        :type ksize: Int
        :param ksize: Window size (odd number)

        :type filter_op: String
        :param filter_op: Filter operation to be used (median/mean)

        :return: Filtered array
        """
        from numpy.lib.stride_tricks import as_strided

        # Available filter operations
        filter_dict = {'median': np.median, 'mean': np.mean}
        filter_op = filter_dict[filter_op]

        # Get window radius, padded array and strides
        ksize += 1 - ksize % 2  # Make sure window size is an odd number
        radius = ksize // 2  # Window radius
        padded = np.pad(in_arr, radius, 'reflect')  # Pad array using the radius
        strides = padded.strides + padded.strides

        # Parameters that depend on the input array
        try:
            assert len(in_arr.shape) == 2
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
            others_shape = (sy + radius, sx + radius, 1, ksize - radius, ksize - radius, bands)

        # Slice the padded array using strides
        up_down = as_strided(padded, shape=up_down_shape, strides=strides)
        left_right = as_strided(padded, shape=left_right_shape, strides=strides)
        rest = as_strided(padded, shape=others_shape, strides=strides)
        self.remove_var(padded)

        # Get the median value of each sub-window, then flatten them
        up_down_meds = np.apply_over_axes(filter_op, up_down, pr_axes).astype(up_down.dtype)
        self.remove_var(up_down)
        left_right_meds = np.apply_over_axes(filter_op, left_right, pr_axes).astype(left_right.dtype)
        self.remove_var(left_right)
        rest_meds = np.apply_over_axes(filter_op, rest, pr_axes).astype(rest.dtype)
        self.remove_var(rest)

        # Compute filter for subwindows
        up_meds = up_down_meds[:-radius, :].reshape(reshape_shape)
        down_meds = up_down_meds[radius:, :].reshape(reshape_shape)
        left_meds = left_right_meds[:, :-radius].reshape(reshape_shape)
        right_meds = left_right_meds[:, radius:].reshape(reshape_shape)
        nw_meds = rest_meds[:-radius, :-radius].reshape(reshape_shape)
        sw_meds = rest_meds[radius:, :-radius].reshape(reshape_shape)
        ne_meds = rest_meds[:-radius, radius:].reshape(reshape_shape)
        se_meds = rest_meds[radius:, radius:].reshape(reshape_shape)

        # Stack the flatten arrays and find where the minimum value is for each pixel
        stacked = np.vstack((up_meds, down_meds, right_meds, left_meds, nw_meds, sw_meds, ne_meds, se_meds))
        self.remove_var((up_meds, down_meds, right_meds, left_meds, nw_meds, sw_meds, ne_meds, se_meds))
        subtr = np.absolute(stacked - in_arr.reshape(reshape_shape))
        self.remove_var(in_arr)
        inds = np.argmin(subtr, axis=0)  # Get indices where the subtr is minimum along the 0 axis
        self.remove_var(subtr)

        # Get the output pixel values
        if not bands:
            filt = np.take_along_axis(stacked, np.expand_dims(inds, axis=0), axis=0).reshape(sy, sx)
        else:
            filt = np.take_along_axis(stacked, np.expand_dims(inds, axis=0), axis=0).reshape(sy, sx, bands)
            # filt = np.flip(filt, 2)  # Flip channel order from BGR to RGB

        self.remove_var(stacked)

        return filt

    """Data Conversion"""

    def resample_array(self, in_arr, out_shape=None, in_pix=None, out_pix=None):
        """
        Method that resamples arrays using the shape or the pixel size.

        :type in_arr: Array
        :param in_arr: 2D/3D Image array

        :type out_shape: Tuple
        :param out_shape: Tuple of output array dimension (e.g. (nrows, ncols))

        :type in_pix: Float/Int
        :param in_pix: Input pixel size. Provide instead of out_shape. For non-square pixels, provide a tuple (psy, psx)

        :type out_pix: Float/Int :param out_pix: Output pixel size. Provide along with in_pix instead of out_shape.
        For non-square pixels, provide a tuple (psy, psx)

        :return: Resampled array, Adjusted Geo-transformation
        """

        # Make sure in_arr is of a supported dtype
        try:
            assert self.__dtype(in_arr) in ('uint8', 'uint16', 'float32')
        except AssertionError:
            in_arr = in_arr.astype(self.__dtype(in_arr)[1])

        # Resize array
        if out_shape:
            if not in_arr.shape == out_shape:
                resampled = cv2.resize(in_arr, out_shape)
            else:
                resampled = in_arr
        elif in_pix and out_pix:
            if not isinstance(in_pix, (list, tuple)) and not isinstance(out_pix, (list, tuple)):
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

    def rgb16to8(self, rgb_stack):
        """
        Method to convert a 16bit RGB to 8bit with contrast enhancement.

        :type rgb_stack: Tuple/List of arrays
        :param rgb_stack: Tuple/List containing the Red, Green and Blue arrays. Must be same shape.

        :return: 8bit RGB array
        """

        # Iterate over each array and get min and max corresponding to a 5-95% data truncation
        # Also, resample if needed to the largest size

        rgb_8bit = rgb_stack.copy()
        self.remove_var(rgb_stack)

        try:
            for i in range(rgb_8bit.shape[2]):
                # Get image statistics
                av_val = np.mean(rgb_8bit[:, :, i])
                std_val = np.std(rgb_8bit[:, :, i])
                min = av_val - 1.96 * std_val
                min *= int(min >= 0)  # make sure min value is not negative
                max = av_val + 1.96 * std_val

                # Truncate the array - Contrast Enhancement
                rgb_8bit[:, :, i][rgb_8bit[:, :, i] > max] = max
                rgb_8bit[:, :, i][rgb_8bit[:, :, i] < min] = min

                # Convert to 8bits
                rgb_8bit[:, :, i] = np.divide(rgb_8bit[:, :, i] - min, max - min) * 255
        except IndexError:
            # Get image statistics
            av_val = np.mean(rgb_8bit)
            std_val = np.std(rgb_8bit)
            min = av_val - 1.96 * std_val
            min *= int(min >= 0)  # make sure min value is not negative
            max = av_val + 1.96 * std_val

            # Truncate the array - Contrast Enhancement
            rgb_8bit[rgb_8bit > max] = max
            rgb_8bit[rgb_8bit < min] = min

            # Convert to 8bits
            rgb_8bit = np.divide(rgb_8bit - min, max - min) * 255

        return rgb_8bit

    """Phase Correlation"""

    @staticmethod
    def estimate_disp(disp_map):
        """
        Method to estimate subpixel displacement.

        :type disp_map: Array
        :param disp_map: Displacement map obtained from the phase_correlation method

        :return: Displacement value
        """

        import math

        nrow, ncol = disp_map.shape  # Get # of rows in correlation surface
        peak_y, peak_x = np.unravel_index(np.argmax(disp_map, axis=None), disp_map.shape)

        # Get displacements adjacent to peak
        x_bef = (peak_x - 1 >= 0) * (peak_x - 1) + (peak_x - 1 < 0) * (peak_x)
        x_aft = (peak_x + 1 >= ncol - 1) * (peak_x) + (peak_x + 1 < ncol - 1) * (peak_x + 1)
        y_bef = (peak_y - 1 >= 0) * (peak_y - 1) + (peak_y - 1 < 0) * (peak_y)
        y_aft = (peak_y + 1 >= nrow - 1) * (peak_y) + (peak_y + 1 < nrow - 1) * (peak_y + 1)

        # Estimate subpixel displacement in x-direction
        dx_num = np.log(disp_map[peak_y, x_aft]) - np.log(disp_map[peak_y, x_bef])
        dx_denom = 2 * (2 * np.log(disp_map[peak_y, peak_x]) - np.log(disp_map[peak_y, x_aft]) - np.log(
            disp_map[peak_y, x_bef]))
        dx = dx_num / dx_denom  # subpixel motion in x direction (East/West)
        if math.isnan(dx):
            dx = 0.0

        # Estimate subpixel displacement in y-direction
        dy_num = np.log(disp_map[y_aft, peak_x]) - np.log(disp_map[y_bef, peak_x])
        dy_denom = 2 * (2 * np.log(disp_map[peak_y, peak_x]) - np.log(disp_map[y_aft, peak_x]) - np.log(
            disp_map[y_bef, peak_x]))
        dy = dy_num / dy_denom  # subpixel motion in y direction (North/South)
        if math.isnan(dy):
            dy = 0.0

        if np.abs(peak_x) > disp_map.shape[1] / 2:
            peak_x = peak_x - disp_map.shape[1] + 1  # convert x offsets > ws/2 to negative offsets
        if np.abs(peak_y) > disp_map.shape[0] / 2:
            peak_y = peak_y - disp_map.shape[0] + 1  # convert y offsets > ws/2 to negative offsets
        disx = peak_x + dx
        disy = peak_y + dy

        dis = np.sqrt(disx ** 2 + disy ** 2)

        return float(dis > 0.71) * dis

    def kernel_disp(self, red_tile, green_tile, blue_tile):
        """
        Method to get the mean displacement within an RGB tile.

        :type red_tile: 2D Array
        :param red_tile: Red band tile

        :type green_tile: 2D Array
        :param green_tile: Green band tile

        :type blue_tile: 2D Array
        :param blue_tile: Blue band tile

        :return: Mean subpixel displacement
        """

        # Phase correlation
        G_r = np.fft.fft2(red_tile)  # FFT
        G_g = np.fft.fft2(green_tile)
        G_b = np.fft.fft2(blue_tile)
        conj_b = np.ma.conjugate(G_b)  # Complex conjugate
        conj_g = np.ma.conjugate(G_g)
        R_rg = G_r * conj_g  # Cross-power spectrum
        R_bg = G_b * conj_g
        R_rb = G_r * conj_b
        R_rg /= np.absolute(R_rg)
        R_bg /= np.absolute(R_bg)
        R_rb /= np.absolute(R_rb)
        disp_map_rg = np.fft.ifft2(R_rg).real  # Normalized cross-correlation
        disp_map_bg = np.fft.ifft2(R_bg).real
        disp_map_rb = np.fft.ifft2(R_rb).real

        # Estimate absolute displacement in pixels for each band pair
        disp_rg = self.estimate_disp(disp_map_rg)
        disp_bg = self.estimate_disp(disp_map_bg)
        disp_rb = self.estimate_disp(disp_map_rb)

        # Estimate weighted mean displacement
        mean_disp = np.mean((disp_bg, disp_rb, disp_rg))

        return mean_disp

    def phase_correlation(self, rgb_stack, ksize=3, transf=(0, 1, 0, 0, 0, -1)):
        """
        Wrapper method to estimate subpixel displacement in an RGB stack.

        :type rgb_stack: 3D Array
        :param rgb_stack: RGB stack array

        :type ksize: Int
        :param ksize: Kernel size

        :type transf: Tuple
        :param transf: Geo-transformation tuple

        :return: Subpixel displacement map,  geo-transformation tuple
        """

        # Get individual bands
        red = rgb_stack[:, :, 0]
        green = rgb_stack[:, :, 0]
        blue = rgb_stack[:, :, 0]

        # Histogram equalization to enhance results
        cv2.equalizeHist(red, red)
        cv2.equalizeHist(green, green)
        cv2.equalizeHist(blue, blue)

        # New geographic transformation
        new_tr = (transf[0], transf[1] * ksize, 0, transf[3], 0, transf[5] * ksize)

        # Get tiles for each band
        red_wins = self.get_tiles(in_arr=red, ksize=ksize)
        green_wins = self.get_tiles(in_arr=green, ksize=ksize)
        blue_wins = self.get_tiles(in_arr=blue, ksize=ksize)

        # Get displacement map
        disp_map = np.array([[self.kernel_disp(red_wins[i][j], green_wins[i][j], blue_wins[i][j])
                             for j in range(blue_wins.shape[1])] for i in range(blue_wins.shape[0])])

        return disp_map, new_tr
