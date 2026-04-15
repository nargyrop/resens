import logging
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

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
    nodata: Number = None


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
            nodata=self.nodata,
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
        interpolation: Literal[
            "near",
            "bilinear",
            "cubic",
            "cubicspline",
            "lanczos",
            "average",
            "rms",
            "mode",
            "max",
            "min",
            "med",
            "q1",
            "q3",
            "sum",
        ] = "bilinear",
        inplace: bool = False,
    ) -> Optional["Image"]:
        """Resample the image array by output shape or output pixel size.

        Exactly one of ``out_shape`` or ``out_pix`` should be provided.

        :param out_shape: Output array shape as ``(nrows, ncols)``.
        :param out_pix: Output pixel size. For square pixels pass a single number.
                        For rectangular pixels pass ``(psx, psy)``.
        :param interpolation: Interpolation method: ``near``, ``bilinear``, ``cubic``,
                              ``cubicspline``, ``lanczos``, ``average``, ``rms``,
                              ``mode``, ``max``, ``min``, ``med``, ``q1``, ``q3``,
                              ``sum``
        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a resampled copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        src_ds = self.write_image("MEM")

        # Warp to in-memory output
        if out_shape is not None:
            resampled_ds: gdal.Dataset = gdal.Warp(
                "",
                src_ds,
                format="MEM",
                dstSRS=f"EPSG:{self.epsg_code}",
                width=out_shape[1],
                height=out_shape[0],
                resampleAlg=interpolation,
            )
        elif out_pix is not None:
            if not isinstance(out_pix, list):
                out_pix = [out_pix, out_pix]
            else:
                if not len(out_pix) == 2:
                    out_pix *= 2

            resampled_ds: gdal.Dataset = gdal.Warp(
                "",
                src_ds,
                format="MEM",
                dstSRS=f"EPSG:{self.epsg_code}",
                xRes=out_pix[0],
                yRes=out_pix[1],
                resampleAlg=interpolation,
            )

        if resampled_ds is None:
            raise RuntimeError("Failed to resample image!")

        # Read resampled data
        resampled = self if inplace else self.copy()
        _arr = resampled_ds.ReadAsArray()
        if _arr.ndim == 3:
            _arr = np.einsum("ijk->jki", _arr)
        _arr = _arr.astype(utils.find_dtype(_arr)[1])
        resampled.array = _arr

        resampled.transformation = list(resampled_ds.GetGeoTransform())
        resampled.projection = resampled_ds.GetProjection()
        resampled.epsg_code = int(
            osr.SpatialReference(wkt=resampled.projection).GetAttrValue("AUTHORITY", 1)
        )
        resampled.metadata = resampled_ds.GetMetadata()

        # Clean up
        src_ds = None
        resampled_ds = None

        return resampled if inplace is False else None

    def reproject(self, target_epsg: int, inplace: bool = False) -> Optional["Image"]:
        """Reproject the image to a target CRS.

        :param target_epsg: Target EPSG code. If it matches the source EPSG, the result
                            will be equivalent to the input.
        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a reprojected copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        src_ds = self.write_image("MEM")

        # Warp to in-memory output
        reproj_ds: gdal.Dataset = gdal.Warp(
            "", src_ds, format="MEM", dstSRS=f"EPSG:{target_epsg}"
        )

        if reproj_ds is None:
            raise RuntimeError(f"Failed to reproject image to EPSG:{target_epsg}")

        # Read reprojected data
        reproj = self if inplace else self.copy()
        _arr = reproj_ds.ReadAsArray()
        if _arr.ndim == 3:
            _arr = np.einsum("ijk->jki", _arr)
        _arr = _arr.astype(utils.find_dtype(_arr)[1])
        reproj.array = _arr

        reproj.transformation = list(reproj_ds.GetGeoTransform())
        reproj.projection = reproj_ds.GetProjection()
        reproj.epsg_code = int(
            osr.SpatialReference(wkt=reproj.projection).GetAttrValue("AUTHORITY", 1)
        )
        reproj.metadata = reproj_ds.GetMetadata()

        # Clean up
        src_ds = None
        reproj_ds = None

        return reproj if inplace is False else None

    def to_8bit(self, inplace: bool = False) -> Optional["Image"]:
        """Convert the image to 8-bit with simple contrast enhancement.

        The conversion uses histogram truncation based on 2nd-98th percentiles.

        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a converted copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """

        def _make_8bit(arr: np.ndarray) -> np.ndarray:
            # Get image statistics
            min_val = np.nanpercentile(arr, 2)
            min_val = min_val if min_val >= 0 else 0
            max_val = np.nanpercentile(arr, 98)

            # Truncate the array - Contrast Enhancement
            arr = np.clip(arr, min_val, max_val)

            # Convert to 8bits
            arr = np.divide(arr - min_val, max_val - min_val) * 255

            return arr.astype(np.uint8)

        conv_8bit = self if inplace else self.copy()
        conv_8bit.array = conv_8bit.array.astype(np.float32)

        if conv_8bit.array.ndim == 3:
            out = np.empty(conv_8bit.array.shape, dtype=np.uint8)
            for i in range(conv_8bit.array.shape[2]):
                out[..., i] = _make_8bit(conv_8bit.array[..., i])
        else:
            out = _make_8bit(conv_8bit.array)

        conv_8bit.array = out
        return conv_8bit if not inplace else None

    def to_grayscale(self, inplace: bool = False) -> Optional["Image"]:
        """
        Method to convert a multiband Image to single band (grayscale).

        :param inplace: If ``True``, modify the current object and return ``None``.
                        If ``False``, return a grayscale copy.
        :return: A new :class:`~resens.base.Image` or ``None`` if ``inplace=True``.
        """
        # If the image is not composed of multiple channels, it is already grayscale
        if self.array.ndim != 3:
            return self.array

        src = self.array  # capture before potential inplace overwrite
        grayscale = self if inplace else self.copy()

        # Use underlying data for weight computation to avoid masked-array division issues
        src_data = src.data if isinstance(src, np.ma.MaskedArray) else src
        img_sum = np.sum(src_data, axis=2)  # H×W sum across bands

        # Compute each channel's mean fractional contribution without a full H×W×C copy
        channel_weights = np.array(
            [
                np.nanmean(np.divide(src_data[..., i], img_sum))
                for i in range(src_data.shape[2])
            ]
        )
        del img_sum
        channel_weights = channel_weights / channel_weights.sum()  # Normalize to sum to 1

        # Accumulate weighted bands one at a time to limit peak memory usage
        result = np.zeros(src_data.shape[:2], dtype=np.float64)
        for i, w in enumerate(channel_weights):
            result += src_data[..., i] * w

        grayscale.array = result

        return grayscale if not inplace else None

    def write_image(
        self,
        path: Union[Path, str, Literal["MEM"]],
        nodata: Number = None,
        compression: bool = True,
        datatype: str = None,
        metadata: Dict = None,
    ) -> Union[gdal.Dataset, None]:
        """Write the image to a GeoTIFF on disk.

        :param path: Output path. If no suffix is provided, ``.tif`` is appended. Set to
            ``"MEM"`` to write the raster to memory.
        :param nodata: NoData value to set on the output band(s), if provided.
        :param compression: If ``True``, enable DEFLATE compression. Not valid if
            ``path="MEM"``.
        :param datatype: Output array datatype. If ``None``, the datatype is inferred.
            Supported: ``uint8``, ``uint16``, ``int8``, ``int16``, ``float32``.
        :param metadata: Extra metadata to write to the output dataset. If provided, it
            is merged with existing ``self.metadata``.
        :return: gdal.Dataset if ``path="MEM"``, otherwise ``None``.
        """
        if nodata is None:
            nodata = self.nodata

        # Check that the pixel sizes are of the correct sign
        if self.transformation[-1] > 0:
            self.transformation[-1] *= -1

        # Get array type
        if datatype is None:
            datatype, _ = utils.find_dtype(self.array)
            self.array = self.array.astype(datatype)

        gdal_datatype = rasteroptions.GDAL_DTYPES[datatype]

        # Determine the shape of the array and the number of bands
        if self.array.ndim == 3:
            nrows, ncols, nband = self.array.shape
        else:
            nrows, ncols = self.array.shape
            nband = 1

        # Construct output image path
        if not path == "MEM":
            if isinstance(path, str):
                path = Path(path)
            if path.suffix == "":
                path = path.with_suffix(".tif")

            driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        else:
            driver: gdal.Driver = gdal.GetDriverByName("MEM")
        dataset: gdal.Dataset = driver.Create(
            path.as_posix() if not path == "MEM" else "",
            ncols,
            nrows,
            nband,
            gdal_datatype,
            options=[]
            if path == "MEM"
            else rasteroptions.CO_COMPRESS
            if compression
            else rasteroptions.CO_NOCOMPRESS,
        )
        dataset.SetGeoTransform(self.transformation)
        dataset.SetProjection(self.projection)
        if metadata is not None:
            dataset.SetMetadata({**self.metadata, **metadata})

        for i in range(nband):
            out_band: gdal.Band = dataset.GetRasterBand(i + 1)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            band_data = self.array[..., i] if nband > 1 else self.array
            out_band.WriteArray(band_data)
            out_band = None

        dataset.FlushCache()
        if path == "MEM":
            return dataset
        else:
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

        return clipped if not inplace else None
