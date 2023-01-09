import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
from osgeo import gdal, osr

from . import rasteroptions, utils


def load_image(
        img_path: Union[Path, str],
        bounds: Tuple = None,
    ) -> Iterable:
    """
    Method to load an array from a raster file and retrieve the geo-
    transformation, projection and EPSG of the CRS.

    :param img_path: Path to image
    :param bounds: Tuple containing bounds to be used for clipping the image. 
    Should be formatted as ((xmin, ymax), (xmax, ymin)).
    :return: tuple containing: Array, geo-transformation, projection, epsg code
    """

    kwargs = {}

    if isinstance(img_path, Path):
        img_path = img_path.as_posix()
    dataset = gdal.Open(img_path)

    # Get geographic information
    transf = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    epsg = srs.GetAttrValue("AUTHORITY", 1)

    # If bounds have been passed, calculate the extents for clipping
    if bounds:
        (xmin, ymax), (xmax, ymin) = bounds
        xo, px, _, yo, _, py = transf
        xoff = int(round((xmin - xo) / px))
        yoff = int(round((ymax - yo) / py))
        xsize = int(round((xmax - xo) / px)) - xoff
        ysize = int(round((ymin - yo) / py)) - yoff

        if xsize < 0 or ysize < 0:
            if px == py:
                raise ValueError("Negative dimensions encountered when cropping. \
                Image transformation is invalid.")
            else:
                raise ValueError("Negative dimensions encountered when cropping. \
                    Check bound coordinates.")

        kwargs["xoff"] = xoff
        kwargs["yoff"] = yoff
        kwargs["xsize"] = xsize
        kwargs["ysize"] = ysize

    # Read array
    array = dataset.ReadAsArray(**kwargs)
    if array.ndim == 3:
        array = np.einsum("ijk->jki", array)
    array = array.astype(utils.find_dtype(array)[1])

    # Check that the pixel sizes are of the correct sign
    xo, psx, skx, yo, sky, psy = list(transf)
    if psy > 0:
        psy *= -1
    if bounds:
        xo = float(xmin)
        yo = float(ymax)
    transf = (xo, psx, skx, yo, sky, psy)

    dataset = None

    return array, transf, proj, epsg

def load_from_zip(
    zipf_path: Union[Path, str],
    req_files: Union[list, tuple],
    extension: str,
    group: str = "",
    bounds: Tuple = None
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
            if f.endswith(extension) and any(x in f for x in req_files) \
                and group in f
        ]

        # Create dictionaries to store the data
        band_dict = {}

        for img in img_ls:
            try:
                # Find which of the req files fits the current, create the dict key
                # and store it in the keys' list
                (key_in,) = [key for key in req_files if key in img]

                if key_in in band_dict:
                    print(f"WARNING: Multiple files were found for key: {key_in}!")
                    continue

                # Load image, get metadata and store to dictionary
                array, transf, proj, epsg = load_image(
                    ziphandler + zipf_path.joinpath(img).as_posix(),
                    bounds
                )
                band_dict[key_in] = [array, transf, proj, epsg]
            except AttributeError:
                raise AttributeError(f"Error loading {img}")

        return band_dict

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

    # Check that the pixel sizes are of the correct sign
    xo, psx, skx, yo, sky, psy = list(transformation)
    if psy > 0:
        psy *= -1
    transformation = (xo, psx, skx, yo, sky, psy)

    # Get array type
    if datatype is None:
        datatype, _ = utils.find_dtype(out_arr)

    out_arr = out_arr.astype(datatype)
    gdal_datatype = rasteroptions.GDAL_DTYPES[datatype]

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
        options=rasteroptions.CO_COMPRESS if compression else rasteroptions.CO_NOCOMPRESS,
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
        out_band = None
    
    dataset = None
