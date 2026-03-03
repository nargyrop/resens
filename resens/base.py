import logging
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from osgeo import gdal, osr
from pyproj import Transformer

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
    def copy(self) -> "Image":
        """Creates a deep copy of the :class:`~resens.base.Image` instance.

        :return: Returns a copy of :class:`~resens.base.Image`
        """
        return Image(
            array=self.array.copy(),
            transformation=self.transformation.copy(),
            projection=self.projection,
            epsg_code=self.epsg_code,
            metadata=self.metadata.copy(),
        )

    def extents(
        self, dst_epsg: Optional[int] = None
    ) -> Tuple[float, float, float, float]:
        """Return the bounding box of a georeferenced image.

        The bounds are derived from the image geo-transform and array shape.
        :param dst_epsg: Pass an EPSG code to transform to another CRS, defaults to None.

        :return: Bounding box as ``(xmin, ymin, xmax, ymax)``.
        """
        shape = self.array.shape[:2]
        xmin, psx, _, ymax, _, psy = self.transformation

        bbox = (
            xmin,
            ymax + shape[0] * psy + psy / 2,
            xmin + shape[1] * psx + psx / 2,
            ymax,
        )

        # optional: transform extents bbox to another CRS using an EPSG code
        if dst_epsg is not None and dst_epsg != self.epsg_code:
            transformer = Transformer.from_crs(
                f"EPSG:{self.epsg_code}",
                f"EPSG:{dst_epsg}",
                always_xy=True,  # ensures consistent (lon, lat) / (x, y) order
            )

            min_x, min_y = transformer.transform(bbox[0], bbox[1])
            max_x, max_y = transformer.transform(bbox[2], bbox[3])

            bbox = (min_x, min_y, max_x, max_y)

        return bbox

    def resample(
        self,
        out_shape: Optional[Sequence[int]] = None,
        out_pix: Optional[Number] = None,
        interpolation: Literal["nearest", "linear", "cubic", "lanczos"] = "linear",
        inplace: bool = False,
    ) -> Optional["Image"]:
        """Resample the image array by output shape or output pixel size.

        Exactly one of ``out_shape`` or ``out_pix`` should be provided.

        :param out_shape: Output array shape as ``(nrows, ncols)``.
        :param out_pix: Output pixel size. For square pixels pass a single number.
                        For rectangular pixels pass ``(psx, psy)``.
        :param interpolation: Interpolation method: ``nearest``, ``linear``, ``cubic``,
                              or ``lanczos``.
        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a resampled copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        inter_method = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "nearest": cv2.INTER_NEAREST,
        }

        resampled = self.copy() if inplace else self

        # Make sure in_arr is of a supported dtype
        try:
            assert utils.find_dtype(resampled.array) in ("uint8", "uint16", "float32")
        except AssertionError:
            resampled.array = resampled.array.astype(utils.find_dtype(resampled.array)[1])

        # Resize array
        if out_shape is not None:
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
        elif out_pix is not None:
            _, psx, _, _, _, psy = resampled.transformation
            if not isinstance(out_pix, (list, tuple)):
                # Square pixels
                scalex = psx / out_pix
                scaley = abs(psy) / out_pix
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
        """Reproject the image to a target CRS.

        :param target_epsg: Target EPSG code. If it matches the source EPSG, the result
                            will be equivalent to the input.
        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a reprojected copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        # Create in-memory dataset for source
        mem_driver = gdal.GetDriverByName("MEM")
        datatype, _ = utils.find_dtype(self.array)
        gdal_datatype = rasteroptions.GDAL_DTYPES[datatype]

        # Determine array dimensions
        if self.array.ndim == 3:
            nrows, ncols, nbands = self.array.shape
        else:
            nrows, ncols = self.array.shape
            nbands = 1

        # Create source dataset in memory
        src_ds = mem_driver.Create("", ncols, nrows, nbands, gdal_datatype)
        src_ds.SetGeoTransform(self.transformation)
        src_ds.SetProjection(self.projection)

        # Write array to source dataset
        if nbands == 1:
            src_ds.GetRasterBand(1).WriteArray(self.array)
        else:
            for i in range(nbands):
                src_ds.GetRasterBand(i + 1).WriteArray(self.array[:, :, i])

        # Flush to ensure data is written
        src_ds.FlushCache()

        # Warp to in-memory output
        reproj_ds = gdal.Warp("", src_ds, format="MEM", dstSRS=f"EPSG:{target_epsg}")

        if reproj_ds is None:
            raise RuntimeError(f"Failed to reproject image to EPSG:{target_epsg}")

        # Read reprojected data
        reproj = self if inplace else self.copy()
        reproj.array = reproj_ds.ReadAsArray()
        if reproj.array.ndim == 3:
            reproj.array = np.einsum("ijk->jki", reproj.array)
        reproj.array = reproj.array.astype(utils.find_dtype(reproj.array)[1])

        reproj.transformation = list(reproj_ds.GetGeoTransform())
        reproj.projection = reproj_ds.GetProjection()
        reproj.epsg_code = int(
            osr.SpatialReference(wkt=reproj.projection).GetAttrValue("AUTHORITY", 1)
        )
        reproj.metadata = reproj_ds.GetMetadata()

        # Clean up
        src_ds = None
        reproj_ds = None

        return reproj if not inplace else None

    def to_8bit(self, inplace: bool = False) -> Optional["Image"]:
        """Convert the image to 8-bit with simple contrast enhancement.

        The conversion uses histogram truncation based on mean ± 1.96 * std.

        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a converted copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """

        def _make_8bit(arr: np.ndarray) -> np.ndarray:
            # Get image statistics
            av_val = np.mean(arr)
            std_val = np.std(arr)
            min_val = av_val - 1.96 * std_val
            min_val = min_val if min_val >= 0 else 0
            max_val = av_val + 1.96 * std_val

            # Truncate the array - Contrast Enhancement
            arr[arr > max_val] = max_val
            arr[arr < min_val] = min_val

            # Convert to 8bits
            arr = np.divide(arr - min_val, max_val - min_val) * 255

            return arr.astype(np.uint8)

        conv_8bit = self if inplace else self.copy()
        bit8_img = conv_8bit.array.astype(np.float32)

        if bit8_img.ndim == 3:
            for i in range(bit8_img.shape[2]):
                bit8_img[:, :, i] = _make_8bit(bit8_img[:, :, i])
        else:
            bit8_img = _make_8bit(bit8_img)

        conv_8bit.array = bit8_img
        return conv_8bit if not inplace else None

    def to_grayscale(self, inplace: bool = False) -> Optional["Image"]:
        """
        Method to convert a multiband Image to single band (grayscale).

        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a grayscale copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        # If the image is not composed of multiple channels, it is already grayscale
        try:
            assert self.array.ndim == 3
        except AssertionError:
            return self.array

        grayscale = self if inplace else self.copy()
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
        """Write the image to a GeoTIFF on disk.

        :param path: Output path. If no suffix is provided, ``.tif`` is appended.
        :param nodata: NoData value to set on the output band(s), if provided.
        :param compression: If ``True``, enable DEFLATE compression.
        :param datatype: Output array datatype. If ``None``, the datatype is inferred.
            Supported: ``uint8``, ``uint16``, ``int8``, ``int16``, ``float32``.
        :param metadata: Extra metadata to write to the output dataset. If provided, it
            is merged with existing ``self.metadata``.
        :return: ``None``.
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

    def imshow(
        self, image_extents: Optional[List] = None, bbox: Optional[List] = None, **kwargs
    ) -> AxesImage:
        """Plots an array with matplotlib. If a 3D array is passed.

        :param image_extents: Image extents in pixels for clipping formatted as (`minx`,
            `miny`, `maxx`, `maxy`), defaults to None. Overrides ``bbox``.
        :param bbox: Image extents in CRS coordinates for clipping formatted as (`minx`,
            `miny`, `maxx`, `maxy`), defaults to None.

        :return: AxesImage object
        """
        plot_img = self.clip(image_extents, bbox, silent=True)
        strtype = utils.find_dtype(plot_img.array)[0]

        if plot_img.array.ndim == 3:
            if plot_img.array.shape[-1] < 3:
                plot_img.array = plot_img.array[..., 0]
            else:
                plot_img.array = plot_img.array[..., :3]
        else:
            plot_img.array = plot_img.array

        if strtype != "uint8":
            ret = plt.imshow(
                np.divide(
                    plot_img.array - plot_img.array.min(),
                    plot_img.array.max() - plot_img.array.min(),
                ),
                **kwargs,
            )
        else:
            ret = plt.imshow(plot_img.array, **kwargs)

        return ret

    def clip(
        self,
        image_extents: Optional[List] = None,
        bbox: Optional[List] = None,
        inplace: bool = False,
        silent: bool = False,
    ) -> Optional["Image"]:
        """Clips an array.

        :param image_extents: Image extents in pixels for clipping formatted as (`minx`,
            `miny`, `maxx`, `maxy`), defaults to None. Overrides ``bbox``.
        :param bbox: Image extents in CRS coordinates for clipping formatted as (`minx`,
            `miny`, `maxx`, `maxy`), defaults to None.

        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        clipped = self if inplace else self.copy()

        if image_extents is not None:
            minx, miny, maxx, maxy = image_extents
        elif bbox is not None:
            g_minx, g_miny, g_maxx, g_maxy = bbox

            minx = max(
                0, (g_minx - clipped.transformation[0]) / clipped.transformation[1]
            )
            maxx = (g_maxx - clipped.transformation[0]) / clipped.transformation[1]
            miny = max(
                0, (g_maxy - clipped.transformation[3]) / clipped.transformation[-1]
            )
            maxy = (g_miny - clipped.transformation[3]) / clipped.transformation[-1]
        else:
            if not silent:
                logger.warning(
                    "``image_extents`` or ``bbox`` need to be passed to clip the image!"
                )
            return clipped

        minx, miny, maxx, maxy = list(
            map(lambda x: int(round(x)), [minx, miny, maxx, maxy])
        )

        # clip image
        clipped.array = clipped.array[miny:maxy, minx:maxx]

        # adjust transformation
        clipped.transformation[0] -= minx * clipped.transformation[1]
        clipped.transformation[3] -= maxy * clipped.transformation[-1]

        return clipped
