import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
from osgeo import gdal, osr

from . import utils
from .base import Image

logger = logging.getLogger(__name__)

__all__ = ["load_image", "load_from_zip"]


def load_image(
    img_path: Union[Path, str],
    bbox: Tuple = None,
    fill_outside: bool = False,
    masked: bool = False,
    **kwargs,
) -> Image:
    """Load a raster file into an :class:`~resens.base.Image`.

    The function reads the array, geo-transform, projection and metadata using GDAL.

    :param img_path: Path to image
    :param bbox: Tuple containing bounding box coordinatee to be used for clipping the
        image. Should be formatted as (xmin, ymin, xmax, ymax).
    :param fill_outside: If ``True``, pixels outside raster coverage (when ``bbox`` exceed
        the raster extent) are padded with zeros.
    :param masked: If ``True`` and the raster has an assigned nodata value, pixels with
        this value will be converted to `nan`. Masking the array will convert it to
        float32, increasing the memory consumption.
    :param kwargs: Optional overrides for ``transformation`` and ``projection``.
    :return: Loaded :class:`~resens.base.Image`.
    """
    load_kwargs = {}

    img_path = img_path.as_posix() if isinstance(img_path, Path) else img_path
    dataset: gdal.Dataset = gdal.Open(img_path)

    # Get geographic information
    transf = kwargs.get("transformation", None) or list(dataset.GetGeoTransform())
    proj = (
        kwargs.get("projection", None)
        or dataset.GetProjection()
        or dataset.GetGCPProjection()
    )

    srs = osr.SpatialReference(wkt=proj)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    srs.AutoIdentifyEPSG()
    epsg = (
        srs.GetAuthorityCode(None)
        or srs.GetAuthorityCode("PROJCS")
        or srs.GetAuthorityCode("GEOGCS")
    )

    # in case the raster is geoereferenced with GCPs, warp it to create a raster with
    # geotransformation
    _vsimem_path = None
    if transf == [0, 1, 0, 0, 0, 1] and len(dataset.GetGCPs()):
        _vsimem_path = "/vsimem/warp.tif"
        dataset = gdal.Warp(_vsimem_path, dataset, dstSRS=f"EPSG:{epsg}")
        transf = list(dataset.GetGeoTransform())

    # Get metadata
    metadata = dataset.GetMetadata()
    nodata_val = dataset.GetRasterBand(1).GetNoDataValue()

    # If bounds have been passed, calculate the extents for clipping
    if bbox:
        raster_size = dataset.RasterXSize, dataset.RasterYSize
        xmin, ymin, xmax, ymax = bbox
        xo, px, _, yo, _, py = transf
        xoff = int(np.ceil((xmin - xo) / px))
        yoff = int(np.ceil((ymax - yo) / py))
        padx = [0, 0]
        pady = [0, 0]
        if fill_outside:
            if xoff < 0:
                padx[0] = abs(xoff)
            if yoff < 0:
                pady[0] = abs(xoff)
        xoff = 0 if xoff < 0 else xoff
        yoff = 0 if yoff < 0 else yoff

        xsize = int(np.floor((xmax - xo) / px)) - xoff
        ysize = int(np.floor((ymin - yo) / py)) - yoff
        if fill_outside:
            if xsize + xoff > raster_size[0]:
                padx[1] = xsize + xoff - raster_size[0]
                xsize -= padx[1]
            if ysize + yoff > raster_size[1]:
                pady[1] = ysize + yoff - raster_size[1]
                ysize -= pady[1]

        if xsize < 0 or ysize < 0:
            if px == py:
                raise ValueError(
                    "Negative dimensions encountered when cropping. "
                    "Image transformation is invalid."
                )
            else:
                raise ValueError(
                    "Negative dimensions encountered when cropping. "
                    "Check bound coordinates."
                )

        load_kwargs["xoff"] = xoff
        load_kwargs["yoff"] = yoff
        load_kwargs["xsize"] = xsize
        load_kwargs["ysize"] = ysize

    # Read array
    array: np.ndarray = dataset.ReadAsArray(**load_kwargs)
    if array.ndim == 3:
        array = np.einsum("ijk->jki", array)

    if nodata_val is not None and masked is True:
        array = array.astype(np.float32)
        array[array == nodata_val] = np.nan
        dtype_s = "float32"
    else:
        dtype_s, dtype_np = utils.find_dtype(array)
        array = array.astype(dtype_np)

    # Pad the array if the selected bounds are outside of its dimensions
    if bbox and fill_outside:
        pad_widths = [pady, padx]

        if array.ndim == 3:
            pad_widths += [[0, 0]]  # 2-axis padding (no padding)
        array = np.pad(array, pad_widths, "constant")

    # Check that the pixel sizes are of the correct sign
    xo, psx, skx, yo, sky, psy = list(transf)
    if psy > 0:
        psy = -psy
    if bbox:
        xo += (xoff - padx[0]) * psx
        yo += (yoff - pady[0]) * psy
    transf = [xo, psx, skx, yo, sky, psy]

    dataset = None
    if _vsimem_path is not None:
        gdal.Unlink(_vsimem_path)

    return Image(array, transf, proj, epsg, metadata, nodata_val)


def load_from_zip(
    zipf_path: Union[Path, str],
    patterns: Union[str, Sequence[str]],
    extension: str,
    group: str = "",
    bounds: Tuple = None,
    fill_outside: bool = False,
    masked: bool = False,
) -> Union[Dict[str, Image], None]:
    """Load rasters from within a ZIP, TAR, or TAR.GZ archive.

    Files are selected by:
    - matching the provided file ``extension``,
    - containing any of the strings in ``patterns``,
    - and containing ``group`` (if provided).

    Supported archive formats: ``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``.

    :param zipf_path: Path to the archive file.
    :param patterns: List of strings included in the file names (e.g. band numbers)
    :param extension: Extension of the target image
    :param group: Extra string to search for.
    :param bounds: Tuple containing bounds to be used for clipping the image.
        Should be formatted as ((xmin, xmax), (xmax, ymin)).
    :param fill_outside: If ``True``, pad outside coverage with zeros when cropping.
    :param masked: If ``True`` and the raster has an assigned nodata value, pixels with
        this value will be converted to `nan`. Masking the array will convert it to
        float32, increasing the memory consumption.
    :return: A dictionary mapping each matched key from ``patterns`` to an
        :class:`~resens.base.Image`. Returns ``None`` if the archive cannot be read.
    """

    # ensure ``patterns`` is a list
    if not isinstance(patterns, (list, tuple)):
        patterns = [patterns]

    # Normalise path
    if isinstance(zipf_path, str):
        zipf_path = Path(zipf_path)
    if not zipf_path.exists():
        raise FileNotFoundError(f"Archive {zipf_path} does not exist!")

    # Choose GDAL virtual-FS handler and Python reader based on file extension
    name_lower = zipf_path.name.lower()
    if name_lower.endswith((".tar.gz", ".tgz", ".tar")):
        vsihandler = "/vsitar/"
        is_tar = True
    else:
        vsihandler = "/vsizip/"
        is_tar = False

    # Open archive and collect matching file names
    try:
        if is_tar:
            archive = tarfile.open(zipf_path, "r:*")
            file_list = archive.getnames()
        else:
            archive = zipfile.ZipFile(zipf_path, "r")
            file_list = archive.namelist()
    except (zipfile.BadZipFile, tarfile.TarError):
        return None

    img_ls = [
        f
        for f in file_list
        if f.endswith(extension) and any(x in f for x in patterns) and group in f
    ]

    band_dict = {}
    for img in img_ls:
        try:
            # Identify which pattern key this file belongs to
            (key_in,) = [key for key in patterns if key in img]

            if key_in in band_dict:
                logger.warning(f"Multiple files were found for key: {key_in}!")
                continue

            band_dict[key_in] = load_image(
                img_path=vsihandler + zipf_path.joinpath(img).as_posix(),
                bbox=bounds,
                fill_outside=fill_outside,
                masked=masked,
            )
        except AttributeError:
            raise AttributeError(f"Error loading {img}")

    return band_dict
