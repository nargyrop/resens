import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import osgeo as _
from resens import io, processing

os.environ["PROJ_LIB"] = Path(_.__path__[0]).joinpath("data/proj").as_posix()
base_path = Path(__file__).parent

class TestSum(unittest.TestCase):
    def test_load_image(self):
        image = io.load_image(
            base_path.joinpath("data", "sample-bgrn-16bit-small.tif")
        )
        self.assertTupleEqual(
            image.array.shape, (1511, 1441, 4), "Array has the correct dimensions"
        )
        self.assertEqual(image.epsg_code, "32639", "The correct EPSG code was loaded")

    def test_write_image(self):
        # First load the sample image
        sample = io.load_image(
            base_path.joinpath("data", "sample-bgrn-16bit-small.tif")
        )

        # Then write a test output image
        output_path = Path(tempfile.gettempdir(), "test_output.tif").as_posix()
        io.write_image(
            out_arr=sample.array,
            output_img=output_path,
            transformation=sample.transformation,
            projection=sample.projection,
            nodata=-1,
            compression=True,
        )

        # Then load the test output
        image = io.load_image(output_path)

        # Now check to make sure everything is correct
        self.assertTrue(np.all(sample.array == image.array), "Arrays are not equal")
        self.assertTupleEqual(sample.transformation, image.transformation, "Transformation is correct")
        self.assertEqual(sample.projection, image.projection, "Projection is correct")
        self.assertEqual(sample.epsg_code, image.epsg_code, "EPSG code is correct")
        self.assertDictEqual(sample.metadata, image.metadata, "Metadata is correct")

    def test_get_sliding_win(self):

        # initialize two random arrays
        arr_sb = np.random.randint(0, 256, (100, 100))
        arr_mb = np.random.randint(0, 256, (100, 100, 3))

        arr_sb_convs = processing.get_sliding_win(
            in_arr=arr_sb, ksize=3, step_x=1, step_y=1, pad=True
        )
        self.assertTupleEqual(
            arr_sb_convs.shape,
            (102, 102, 3, 3),
            "Correct convolution number (singleband)",
        )

        arr_mb_convs = processing.get_sliding_win(
            in_arr=arr_mb, ksize=3, step_x=1, step_y=1, pad=True
        )
        self.assertTupleEqual(
            arr_mb_convs.shape,
            (102, 102, 1, 3, 3, 3),
            "Correct tile number (multiband)",
        )

    def test_get_tiles(self):

        # initialize two random arrays
        arr_sb = np.random.randint(0, 256, (100, 100))
        arr_mb = np.random.randint(0, 256, (100, 100, 3))

        arr_sb_tiles = processing.get_tiles(
            in_arr=arr_sb,
            ksize=3,
        )
        self.assertTupleEqual(
            arr_sb_tiles.shape, (34, 34, 3, 3), "Correct tile number (singleband)"
        )

        arr_mb_tiles = processing.get_tiles(
            in_arr=arr_mb,
            ksize=3,
        )
        self.assertTupleEqual(
            arr_mb_tiles.shape, (34, 34, 1, 3, 3, 3), "Correct tile number (multiband)"
        )


if __name__ == "__main__":
    unittest.main()
