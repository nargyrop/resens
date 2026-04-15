## Changelog

All notable changes to `resens` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [1.2.0] - 2026-04-15

- Added functionality to `get_tiles` and `get_sliding_win` to enable returning georeferenced tiles when
    a `resens.Image` object is passed for tiling
- Moved `get_tiles`, `get_sliding_win` and `swf` to utils
- Extended `load_from_zip` to support `.tar` and `.tar.gz` archives in addition to `.zip`
- Fixed `write_image` nodata bug where `nodata=0` was silently ignored
- Fixed `to_8bit` returning `float32` instead of `uint8` for multiband images
- Fixed `clip` not returning `None` when `inplace=True`
- Fixed `get_tiles` over-padding arrays already divisible by tile size
- Fixed virtual memory leak in `load_image` when loading GCP-georeferenced rasters
- Fixed `copy()` not preserving the `nodata` value
- Fixed `find_dtype` raising errors on empty arrays
- Improved `to_grayscale` memory usage by removing intermediate full-array allocation
- Removed `dtype` attribute from `Image` dataclass (was unused after dtype inference was moved to `find_dtype`)
- Removed `[project]` table from `pyproject.toml` to fix build failures on Python 3.8
- Significantly expanded test suite coverage

### [1.1.1] - 2026-03-18

- Changed the `resample` method to use the GDAL warp command
- Added functionality to enable masking numpy arrays when a nodata value is defined
- Other minor changes that improve functionality

### [1.1.0] - 2026-03-12

- Enabled writing rasters to memory
- Simplified the `reproject` method
- Added functionality to mask nodata values and convert them to nan when using `load_image` with the argument `masked=True`
- Changed the `to_8bit` method to use the 2nd and 98th percentiles for clipping the data

### [1.0.2] - 2026-03-03

- Minor bug fixes

### [1.0.1] - 2026-03-02

- Initial release on PyPI.
- Core raster processing utilities for remote sensing and earth observation.

[1.0.0]: https://github.com/nargyrop/resens/releases/tag/v1.0.0
