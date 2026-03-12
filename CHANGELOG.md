## Changelog

All notable changes to `resens` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [1.0.3] - 2026-03-12

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
