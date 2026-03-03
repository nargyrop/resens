import logging
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from osgeo import gdal, osr

from . import rasteroptions, utils

logger = logging.getLogger(__name__)

__all__ = ["Image"]


@dataclass
class _Image:
    array: np.ndarray = field(default_factory=lambda: np.ndarray([]))
    transformation: List = field(default_factory=lambda: [0, 1, 0, 0, 0, -1])
    projection: str = ""
    epsg_code: int = 1
    metadata: Dict = field(default_factory=dict)


class Image(_Image):
    def __copy__(self):
        return Image(
            array=self.array.copy(),
            transformation=self.transformation.copy(),
            projection=self.projection,
            epsg_code=self.epsg_code,
            metadata=self.metadata.copy(),
        )

    def bounds(self) -> Tuple:
        """Method to the bounds of a georeferenced Image

        :return: Bounding box
        """
        shape = self.array.shape[:2]
        xmin, psx, _, ymax, _, psy = self.transformation

        return xmin, ymax + shape[0] * psy, xmin + shape[1] * psx, ymax

    def resample_array(
        self,
        out_shape: Optional[Sequence[int]] = None,
        out_pix: Optional[Number] = None,
        interpolation: Literal["nearest", "linear", "cubic", "lanczos"] = "linear",
        inplace: bool = False,
    ) -> Optional["Image"]:
        """
        Method that resamples arrays using the shape or the pixel size.

        :param out_shape: Tuple of output array dimension (e.g. (nrows, ncols))
        :param out_pix: Output pixel size. Provide along with in_pix instead of out_shape.
        For non-square pixels, provide a tuple (psx, psy)
        :param interpolation: Interpolation method. Choose between nearest, linear, cubic,
        lanczos.
        :param inplace: If False, return a copy. Otherwise, do operation in place and
        return None.

        :return: Image or None if inplace=True.
        """

        inter_method = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
        }

        resampled = self.__copy__() if inplace else self

        # Make sure in_arr is of a supported dtype
        try:
            assert utils.find_dtype(resampled.array) in ("uint8", "uint16", "float32")
        except AssertionError:
            resampled.array = resampled.array.astype(utils.find_dtype(resampled.array)[1])

        # Resize array
        if out_shape:
            if not resampled.array.shape == out_shape:
                scalex = float(out_shape[1]) / resampled.array.shape[1]
                scaley = float(out_shape[0]) / resampled.array.shape[0]
                resampled.array = cv2.resize(
                    resampled.array,
                    out_shape[::-1],
                    interpolation=inter_method[interpolation],
                )
                resampled.transformation[1] /= scalex
                resampled.transformation[-1] /= scaley
        elif out_pix:
            _, psx, _, _, _, psy = resampled.transformation
            if not isinstance(out_pix, (list, tuple)):
                # Square pixels
                scale = psx / out_pix
                resampled.array = cv2.resize(
                    resampled.array,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=inter_method[interpolation],
                )
                resampled.transformation[1] /= scale
                resampled.transformation[-1] /= scale
            else:
                # Rectangular pixels
                scalex = psx / out_pix[0]
                scaley = abs(psy) / out_pix[1]
                resampled.array = cv2.resize(
                    resampled.array,
                    None,
                    fx=scalex,
                    fy=scaley,
                    interpolation=inter_method[interpolation],
                )
                resampled.transformation[1] /= scalex
                resampled.transformation[-1] /= scaley

        return resampled if not inplace else None

    def reproject(self, target_epsg: int, inplace: bool = False) -> Optional["Image"]:
        """Reproject an Image using an EPSG code.

        :param target_epsg: Target EPSG code. If the target code is the same as the
        source code, the image won't be reprojected.
        :type target_epsg: int
        :param inplace: If False, return a copy. Otherwise, do operation in place and
        return None.

        :return: Image or None if inplace=True.
        """
        temp_input = NamedTemporaryFile(suffix=".tif").name
        temp_output = NamedTemporaryFile(suffix=".tif").name
        self.write_image(temp_input, metadata=self.metadata)
        input_raster = gdal.Open(temp_input)

        warp = gdal.Warp(
            temp_output, input_raster, dstSRS=f"EPSG:{target_epsg}"
        )  # noqa: F841
        input_raster = None
        warp = None  # noqa: F841

        reproj_ds = gdal.Open(temp_output)

        reproj = self.__copy__() if inplace else self
        reproj.array = reproj_ds.ReadAsArray()
        if reproj.array.ndim == 3:
            reproj.array = np.einsum("ijk->jki", reproj.array)
        reproj.array = reproj.array.astype(utils.find_dtype(reproj.array)[1])

        reproj.transformation = list(reproj_ds.GetGeoTransform())
        reproj.projection = reproj_ds.GetProjection()
        reproj.epsg = osr.SpatialReference(wkt=reproj.projection).GetAttrValue(
            "AUTHORITY", 1
        )
        reproj.metadata = reproj_ds.GetMetadata()

        # remove temporary files
        for i in [temp_input, temp_output]:
            Path(i).unlink()

        return reproj if not inplace else None

    def to_8bit(self, inplace: bool = False) -> Optional["Image"]:
        """
        Method to convert any n-Bit image to 8-bit with contrast enhancement
        (histogram truncation).

        :param inplace: If False, return a copy. Otherwise, do operation in place and
        return None.

        :return: Image or None if inplace=True.
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

            return arr.astype(np.uint8)

        conv_8bit = self.__copy__() if inplace else self
        # Iterate over each array and get min and max corresponding to a 5-95% data
        # truncation
        bit8_img = conv_8bit.array.astype(np.float32)

        if bit8_img.ndim == 3:
            for i in range(bit8_img.shape[2]):
                bit8_img[:, :, i] = _make_8bit(bit8_img[:, :, i])
        else:
            # Convert to 8bits
            bit8_img = _make_8bit(bit8_img)
        conv_8bit.array = bit8_img

        return conv_8bit if not inplace else None

    def to_grayscale(self, inplace: bool = False) -> Optional["Image"]:
        """
        Method to convert a multiband Image to single band (grayscale).

        :param inplace: If False, return a copy. Otherwise, do operation in place and
        return None.

        :return: Image or None if inplace=True.
        """

        # If the image is not composed of multiple channels, it is already grayscale
        try:
            assert self.array.ndim == 3
        except AssertionError:
            return self.array

        grayscale = self.__copy__() if inplace else self
        img_sum = np.sum(self.array, axis=2)  # Sum of all channel pixels

        # Compute weight for each channel
        img_weights_arr = np.dstack(
            [np.divide(self.array[..., i], img_sum) for i in range(self.array.shape[2])]
        )
        channel_weights = np.nanmean(img_weights_arr, axis=(0, 1))
        while channel_weights.sum() > 1.0:
            channel_weights -= channel_weights * 0.01  # Make sure the weights sum to 1.0

        # Get grayscale image
        grayscale.array = np.sum(self.array * channel_weights, axis=2)

        return grayscale if not inplace else None

    def write_image(
        self,
        path: Union[Path, str],
        nodata: Number = None,
        compression: bool = True,
        datatype: str = None,
        metadata: Dict = None,
    ):
        """
        Method that writes an array to a georeferenced GeoTIFF file.

        :param path: Output image path.
        :param nodata: NoData value.
        :param compression: True to enable compression (default), False to disable.
        :param datatype: Array datatype. Set to None to have the script automatically
        detect the datatype or select
        between uint8, uint16, int8, int16, float32.
        :param metadata: Dictionary containing metadata that should be written to the
        output image.
        """

        # Check that the pixel sizes are of the correct sign
        if self.transformation[-1] > 0:
            self.transformation[-1] *= -1

        # Get array type
        if datatype is None:
            datatype, _ = utils.find_dtype(self.array)

        self.array = self.array.astype(datatype)
        gdal_datatype = rasteroptions.GDAL_DTYPES[datatype]

        try:
            # Determine the shape of the array and the number of bands
            if self.array.shape[0] > self.array.shape[2]:
                row_ind = 0
                col_ind = 1
                nband = self.array.shape[2]
            else:
                row_ind = 1
                col_ind = 2
                nband = self.array.shape[0]

        except IndexError:
            row_ind = 0
            col_ind = 1
            nband = 1

        # Construct output image path
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".tif")

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            path.as_posix(),
            self.array.shape[col_ind],
            self.array.shape[row_ind],
            nband,
            gdal_datatype,
            options=rasteroptions.CO_COMPRESS
            if compression
            else rasteroptions.CO_NOCOMPRESS,
        )
        dataset.SetGeoTransform(self.transformation)
        dataset.SetProjection(self.projection)
        if metadata is not None:
            dataset.SetMetadata({**self.metadata, **metadata})

        for i in range(nband):
            if not nband == 1:
                out_band = dataset.GetRasterBand(i + 1)
                if nodata:
                    out_band.SetNoDataValue(nodata)
                out_band.WriteArray(self.array[..., i])
            else:
                out_band = dataset.GetRasterBand(i + 1)
                if nodata:
                    out_band.SetNoDataValue(nodata)
                out_band.WriteArray(self.array)
            out_band = None

        dataset = None
