import logging
import tempfile
import uuid
from numbers import Number
from pathlib import Path
from typing import Iterable, Union

import cv2
import geopandas as gpd
import numpy as np
from osgeo import gdal, gdalconst, ogr

from . import io
from .base import Image

logger = logging.getLogger(__name__)

__all__ = ["shapefile_masking"]


def shapefile_masking(
    gdf: gpd.GeoDataFrame,
    shape: Iterable[int],
    transformation: tuple,
    projection: str,
    mask_outpath: Union[Path, str] = None,
    burn_value: Number = 1,
    dilation: bool = False,
    dilation_iters: int = None,
    compression: bool = True,
) -> Image:
    """Rasterize polygons from a GeoDataFrame into a raster mask.

    :param gdf: Polygons as a :class:`geopandas.GeoDataFrame`.
    :param shape: Output array shape, e.g. ``(rows, cols)`` or ``(rows, cols, bands)``.
    :param transformation: GDAL geo-transform tuple/list.
    :param projection: CRS WKT string.
    :param mask_outpath: Optional path for the output mask GeoTIFF. If not provided, a
                         temporary file is used.
    :param burn_value: Value to burn into polygons.
    :param dilation: If ``True``, dilate the mask after rasterization.
    :param dilation_iters: Number of dilation iterations (defaults to 1 if
        ``dilation=True``).
    :param compression: If ``True``, enable GeoTIFF compression.
    :return: Mask as an :class:`~resens.base.Image`.
    """
    # Set up the output filename in a way that it won't be needed to create a
    # mask for arrays with the same extents
    remove_files = []
    out_epsg = gdf.crs.from_wkt(projection).to_epsg()
    if gdf.crs.to_epsg() != out_epsg:
        gdf = gdf.to_crs(epsg=out_epsg)
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
    mask = Image(
        np.zeros(shape[:2], dtype=np.int8),
        transformation,
        projection,
    )
    mask.write_image(mask_outpath, compression)
    target_ds = gdal.Open(mask_outpath.as_posix(), gdalconst.GA_Update)

    # Load shapefile layers
    shp_ds = ogr.Open(polygon_shp.as_posix())
    shp_lyr = shp_ds.GetLayer()

    # Rasterization
    gdal.RasterizeLayer(
        target_ds, [1], shp_lyr, burn_values=[burn_value], options=["ALL_TOUCHED=TRUE"]
    )
    target_ds = None

    # Load the mask array
    mask = io.load_image(mask_outpath)

    # Dilate the mask
    if dilation:
        if dilation_iters:
            mask.array = cv2.dilate(
                mask.array,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                iterations=int(dilation_iters),
            )
        else:
            mask.array = cv2.dilate(
                mask.array,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3)),
                iterations=1,
            )
        mask.write_image(mask_outpath, compression=compression)

    for fil in remove_files:
        try:
            fil_path = Path(fil)
            if fil_path.is_file():
                fil_path.unlink()
            elif fil_path.is_dir():
                for child in fil_path.iterdir():
                    child.unlink()
                fil_path.rmdir()
        except Exception:
            # Best-effort cleanup; ignore errors
            pass

    return mask
