import logging
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
    img_path: Union[Path, str], bounds: Tuple = None, fill_outside: bool = False, **kwargs
) -> Image:
    """
    Method to load an array from a raster file and retrieve the geo-
    transformation, projection and EPSG of the CRS.

    :param img_path: Path to image
    :param bounds: Tuple containing bounds to be used for clipping the image.
    Should be formatted as (xmin, ymin, xmax, ymax).
    :param fill_outside: When the defined bounds are outside of the array's shape, the
    function will fill the pixels outside of the array dimensions with zeroes.
    :return: Named tuple containing: Array, geo-transformation, projection, epsg code,
    dictionary containing selected image metadata
    """

    load_kwargs = {}

    img_path = img_path.as_posix() if isinstance(img_path, Path) else img_path
    dataset = gdal.Open(img_path)

    # Get geographic information
    transf = kwargs.get("transformation", None) or list(dataset.GetGeoTransform())
    proj = kwargs.get("projection", None) or dataset.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    epsg = srs.GetAttrValue("AUTHORITY", 1)

    # Get metadata
    metadata = dataset.GetMetadata()

    # If bounds have been passed, calculate the extents for clipping
    if bounds:
        raster_size = dataset.RasterXSize, dataset.RasterYSize
        xmin, ymin, xmax, ymax = bounds
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
    array = dataset.ReadAsArray(**load_kwargs)
    if array.ndim == 3:
        array = np.einsum("ijk->jki", array)
    array = array.astype(utils.find_dtype(array)[1])

    # Pad the array if the selected bounds are outside of its dimensions
    if bounds and fill_outside:
        pad_widths = [pady, padx]

        if array.ndim == 3:
            pad_widths += [[0, 0]]  # 2-axis padding (no padding)
        array = np.pad(array, pad_widths, "constant")

    # Check that the pixel sizes are of the correct sign
    xo, psx, skx, yo, sky, psy = list(transf)
    if psy > 0:
        psy = -psy
    if bounds:
        xo += (xoff - padx[0]) * psx
        yo += (yoff - pady[0]) * psy
    transf = [xo, psx, skx, yo, sky, psy]

    dataset = None

    return Image(array, transf, proj, epsg, metadata)


def load_from_zip(
    zipf_path: Union[Path, str],
    req_files: Sequence[str],
    extension: str,
    group: str = "",
    bounds: Tuple = None,
    fill_outside: bool = False,
) -> Union[Dict, None]:
    """
    Method that loads all the required bands in arrays and saves them to a
    dictionary.

    :param zipf_path: Path to zip file
    :param req_files: List of strings included in the file names (e.g. band numbers)
    :param extension: Extension of the target image
    :param group: Extra string to search for.
    :param bounds: Tuple containing bounds to be used for clipping the image.
    Should be formatted as ((xmin, xmax), (xmax, ymin)).
    :param fill_outside: When the defined bounds are outside of the array's shape, the
    function will fill the pixels outside of the array dimensions with zeroes.
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
            if f.endswith(extension) and any(x in f for x in req_files) and group in f
        ]

        # Create dictionaries to store the data
        band_dict = {}

        for img in img_ls:
            try:
                # Find which of the req files fits the current, create the dict key
                # and store it in the keys' list
                (key_in,) = [key for key in req_files if key in img]

                if key_in in band_dict:
                    logger.warning(f"Multiple files were found for key: {key_in}!")
                    continue

                # Load image, get metadata and store to dictionary
                band_dict[key_in] = load_image(
                    img_path=ziphandler + zipf_path.joinpath(img).as_posix(),
                    bounds=bounds,
                    fill_outside=fill_outside,
                )
            except AttributeError:
                raise AttributeError(f"Error loading {img}")

        return band_dict
