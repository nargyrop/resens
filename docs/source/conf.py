from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure the project root is on sys.path so autodoc can import `resens`
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "resens"
author = "resens contributors"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# If importing fails on machines without heavy deps (GDAL, cv2), mock them:
autodoc_mock_imports = [
    "cv2",
    "osgeo",
    "osgeo.gdal",
    "osgeo.osr",
    "osgeo.ogr",
    "osgeo.gdalconst",
    "geopandas",
]
