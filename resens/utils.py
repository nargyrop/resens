import tempfile
import uuid
from pathlib import Path
from typing import Tuple, Type, Union

import cv2
import geopandas as gpd
import numpy as np
from numpy import int8, short, single, uint8, ushort
from osgeo import gdal, ogr, gdalconst

from . import io


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

def shapefile_masking(
    polygon_shp: str,
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
    :param shape: Input array's size including channels (e.g. (640, 480, 3))
    :param transformation: Geographic transformation tuple
    :param projection: CRS Projection
    :param mask_outpath: Absolute path (including filename) to output raster mask
    :param burn_value: Value to burn to output raster
    :param dilation: Flag to enable dilating the land mask
    :param dilation_iters: Number of dilation iterations
    :param compression: True to enable compression (default), False to disable.
    :return: Binary mask array
    """
    # Set up the output filename in a way that it won't be needed to create a
    # mask for arrays with the same extents
    remove_files = []
    if "s3" in polygon_shp:
        gdf = gpd.read_file(polygon_shp)

        # Write temporary file
        polygon_shp = tempfile.NamedTemporaryFile().name
        gdf.to_file(polygon_shp)
        remove_files.append(polygon_shp)
    elif not Path(polygon_shp).exists():
        raise FileNotFoundError(f"Polygon shapefile does not exist at {polygon_shp.as_posix()}.")
    
    polygon_shp = Path(polygon_shp)

    # Make sure the shapefile and image are using the same CRS
    gdf = gpd.read_file(polygon_shp)
    out_epsg = gdf.crs.from_wkt(projection).to_epsg()
    if gdf.crs.to_epsg() != out_epsg:
        gdf = gdf.to_crs(epsg=out_epsg)
        
        # Write temporary file
        polygon_shp = Path(tempfile.NamedTemporaryFile().name)
        gdf.to_file(polygon_shp)
        remove_files.append(polygon_shp)

    if mask_outpath:
        mask_outpath = Path(mask_outpath)
        if mask_outpath.suffix != ".tif":
            mask_outpath = mask_outpath.joinpath(f"land_mask_{str(uuid.uuid4())}.tif")
    else:
        mask_outpath = Path(tempfile.NamedTemporaryFile().name).with_suffix(".tif")
        remove_files.append(mask_outpath)
    mask_outpath.parent.mkdir(exist_ok=True, parents=True)

    # Write empty raster and load the dataset
    mask_arr = np.zeros(shape[:2], dtype=np.int8)
    io.write_image(
        mask_arr,
        mask_outpath.as_posix(),
        transformation,
        projection,
        compression,
    )
    target_ds = gdal.Open(mask_outpath.as_posix(), gdalconst.GA_Update)

    # Load shapefile layers
    shp_ds = ogr.Open(polygon_shp.as_posix())
    shp_lyr = shp_ds.GetLayer()

    # Rasterization
    gdal.RasterizeLayer(target_ds, [1], shp_lyr, burn_values=[burn_value])
    target_ds = None

    # Load the mask array
    mask_arr, transf, proj, _ = io.load_image(mask_outpath.as_posix())

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
        io.write_image(mask_arr, mask_outpath.as_posix(), transf, proj)
    
    for fil in remove_files:
        try:
            if fil.is_file():
                mask_outpath.unlink()
            else:
                for child in fil.iterdir():
                    child.unlink()
                fil.rmdir()
        except Exception:
            pass
    
    return mask_arr
