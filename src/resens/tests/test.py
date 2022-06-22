import sys
import tempfile
import unittest
from ast import Mod
from pathlib import Path

import numpy as np
from resens.rasterlib import IO, Processing


class TestSum(unittest.TestCase):
    def test_load_image(self):
        arr, transf, proj, epsg = IO().load_image(
            "src/resens/tests/data/sample-bgrn-16bit-small.tif"
        )
        self.assertTupleEqual(
            arr.shape, (1511, 1441, 4), "Array has the correct dimensions"
        )
        self.assertEqual(epsg, "32639", "The correct EPSG code was loaded")

    def test_write_image(self):
        # First load the sample image
        (arr_sample, transf_sample, proj_sample, epsg_sample) = IO.load_image(
            "src/resens/tests/data/sample-bgrn-16bit-small.tif"
        )

        # Then write a test output image
        output_path = Path(tempfile.gettempdir(), "test_output.tif").as_posix()
        IO().write_image(
            out_arr=arr_sample,
            output_img=output_path,
            transformation=transf_sample,
            projection=proj_sample,
            nodata=-1,
            compression=True,
        )

        # Then load the test output
        arr_test, transf_test, proj_test, epsg_test = IO().load_image(output_path)

        # Now check to make sure everything is correct
        self.assertTrue(np.all(arr_sample == arr_test), "Arrays are not equal")
        self.assertTupleEqual(transf_sample, transf_test, "Transformation is correct")
        self.assertEqual(proj_sample, proj_test, "Projection is correct")
        self.assertEqual(epsg_sample, epsg_test, "EPSG code is correct")

    def test_get_sliding_win(self):

        # initialize two random arrays
        arr_sb = np.random.randint(0, 256, (100, 100))
        arr_mb = np.random.randint(0, 256, (100, 100, 3))

        arr_sb_convs = Processing().get_sliding_win(
            in_arr=arr_sb, ksize=3, step_x=1, step_y=1, pad=True
        )
        self.assertTupleEqual(
            arr_sb_convs.shape,
            (102, 102, 3, 3),
            "Correct convolution number (singleband)",
        )

        arr_mb_convs = Processing().get_sliding_win(
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

        arr_sb_tiles = Processing().get_tiles(
            in_arr=arr_sb,
            ksize=3,
        )
        self.assertTupleEqual(
            arr_sb_tiles.shape, (33, 33, 3, 3), "Correct tile number (singleband)"
        )

        arr_mb_tiles = Processing().get_tiles(
            in_arr=arr_mb,
            ksize=3,
        )
        self.assertTupleEqual(
            arr_mb_tiles.shape, (33, 33, 1, 3, 3, 3), "Correct tile number (multiband)"
        )


if __name__ == "__main__":
    unittest.main()
