try:
    import gdal
except ImportError:
    from osgeo import gdal

CO_COMPRESS = [
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "ZLEVEL=9",
    "NUM_THREADS=ALL_CPUS",
    "BIGTIFF=YES",
]

CO_NOCOMPRESS = ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES"]

GDAL_DTYPES = {
    "uint8": gdal.GDT_Byte,
    "uint16": gdal.GDT_UInt16,
    "int8": gdal.GDT_Byte,
    "int16": gdal.GDT_Int16,
    "float32": gdal.GDT_Float32,
}
