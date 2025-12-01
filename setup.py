from pathlib import Path
from typing import List

from setuptools import setup

from resens.__version__ import __version__

BASE_DIR = Path(__file__).parent
long_description = (BASE_DIR / "README.md").read_text()


def get_required() -> List[str]:
    """Function to get package dependencies

    :return: List of package dependencies
    """

    required = []
    with open(BASE_DIR / "requirements/base.txt", encoding="UTF-8") as reqs:
        for line in reqs:
            line = line.strip()
            if not line:
                continue
            if line[0] not in {"-", "#"}:
                required.append(line)

    return required


setup(
    name="resens",
    version=__version__,
    description="Raster Processing package for Remote Sensing and Earth Observation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nargyrop.com",
    author="Nikos Argyropoulos",
    author_email="n.argiropgeo@gmail.com",
    license="MIT",
    packages=["resens"],
    package_dir={"resens": "resens"},
    python_requires=">=3.8",
    zip_safe=False,
    install_requires=get_required(),
)
