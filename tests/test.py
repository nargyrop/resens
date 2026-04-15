import copy
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry as sg

import resens as rs
from resens import utils
from resens.vectors import shapefile_masking

base_path = Path(__file__).parent
SAMPLE_TIF = base_path / "data" / "sample-bgrn-16bit-small.tif"


class TestLoadImage(unittest.TestCase):
    def test_load_image(self):
        image = rs.load_image(SAMPLE_TIF)
        self.assertTupleEqual(
            image.array.shape, (1511, 1441, 4), "Array has the correct dimensions"
        )
        self.assertEqual(image.epsg_code, "32639", "The correct EPSG code was loaded")

    def test_load_sets_metadata(self):
        image = rs.load_image(SAMPLE_TIF)
        self.assertIsInstance(image.metadata, dict)
        self.assertIsInstance(image.transformation, list)
        self.assertEqual(len(image.transformation), 6)

    def test_load_with_bbox(self):
        full = rs.load_image(SAMPLE_TIF)
        xmin, ymin, xmax, ymax = full.extents()
        xmid = xmin + (xmax - xmin) / 2
        ymid = ymin + (ymax - ymin) / 2
        clipped = rs.load_image(SAMPLE_TIF, bbox=(xmin, ymid, xmid, ymax))
        self.assertLess(clipped.array.shape[0], full.array.shape[0])
        self.assertLess(clipped.array.shape[1], full.array.shape[1])

    def test_load_masked_nodata_pixels_are_masked(self):
        # Write a synthetic image with known nodata pixels and verify they are masked
        sample = rs.load_image(SAMPLE_TIF)
        arr = sample.array.copy()
        arr[0, 0] = 0  # force a zero pixel that will be written as nodata
        synthetic = rs.Image(
            array=arr,
            transformation=sample.transformation,
            projection=sample.projection,
            epsg_code=sample.epsg_code,
        )
        tmp = Path(tempfile.gettempdir()) / "test_nodata_masked_pixels.tif"
        synthetic.write_image(tmp, nodata=0)
        masked = rs.load_image(tmp, masked=True)
        # nodata pixels must become NaN and the array must be promoted to float32
        self.assertEqual(masked.array.dtype, np.float32)
        self.assertTrue(np.any(np.isnan(masked.array)))

    def test_load_masked_no_nodata_unchanged(self):
        # The sample has no nodata — masked=True should not convert to float32
        image = rs.load_image(SAMPLE_TIF, masked=True)
        self.assertIsNone(image.nodata)
        # dtype is inferred from values (16-bit data → uint16)
        self.assertEqual(image.array.dtype, np.uint16)

    def test_fill_outside_extending_left_and_above(self):
        # Bbox extending left and above the raster triggers the pad-at-start branches
        full = rs.load_image(SAMPLE_TIF)
        xmin, ymin, xmax, ymax = full.extents()
        psx = full.transformation[1]
        psy = abs(full.transformation[5])
        # Push the bbox 10 pixels left and 10 pixels above
        extended = (xmin - 10 * psx, ymin, xmax, ymax + 10 * psy)
        padded = rs.load_image(SAMPLE_TIF, bbox=extended, fill_outside=True)
        self.assertGreater(padded.array.shape[1], full.array.shape[1])
        self.assertGreater(padded.array.shape[0], full.array.shape[0])
        # The padded leading columns should be zero
        self.assertEqual(int(padded.array[:, 0, :].sum()), 0)

    def test_fill_outside_extending_right_and_below(self):
        # Bbox extending right and below the raster triggers the pad-at-end branches
        full = rs.load_image(SAMPLE_TIF)
        xmin, ymin, xmax, ymax = full.extents()
        psx = full.transformation[1]
        psy = abs(full.transformation[5])
        extended = (xmin, ymin - 10 * psy, xmax + 10 * psx, ymax)
        padded = rs.load_image(SAMPLE_TIF, bbox=extended, fill_outside=True)
        self.assertGreater(padded.array.shape[1], full.array.shape[1])
        self.assertGreater(padded.array.shape[0], full.array.shape[0])
        # The padded trailing columns should be zero
        self.assertEqual(int(padded.array[:, -1, :].sum()), 0)

    def test_invalid_bbox_outside_raster_raises(self):
        # Bbox entirely to the left of the raster → negative xsize → ValueError
        full = rs.load_image(SAMPLE_TIF)
        xmin, ymin, xmax, ymax = full.extents()
        psx = full.transformation[1]
        out_of_range = (xmin - 2000 * psx, ymin, xmin - 1000 * psx, ymax)
        with self.assertRaises(ValueError):
            rs.load_image(SAMPLE_TIF, bbox=out_of_range)


class TestWriteImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_write_and_reload(self):
        output_path = Path(tempfile.gettempdir(), "test_output.tif")
        self.sample.write_image(path=output_path, compression=True)
        image = rs.load_image(output_path)
        self.assertTrue(np.all(self.sample.array == image.array))
        self.assertListEqual(self.sample.transformation, image.transformation)
        self.assertEqual(self.sample.projection, image.projection)
        self.assertEqual(self.sample.epsg_code, image.epsg_code)
        self.assertDictEqual(self.sample.metadata, image.metadata)

    def test_write_to_mem_returns_dataset(self):
        ds = self.sample.write_image("MEM")
        self.assertIsNotNone(ds)
        self.assertEqual(ds.RasterCount, 4)
        self.assertEqual(ds.RasterYSize, 1511)
        self.assertEqual(ds.RasterXSize, 1441)

    def test_write_no_suffix_appends_tif(self):
        output_path = Path(tempfile.gettempdir()) / "test_no_suffix"
        self.sample.write_image(path=output_path, compression=False)
        self.assertTrue(output_path.with_suffix(".tif").exists())

    def test_write_with_nodata(self):
        output_path = Path(tempfile.gettempdir()) / "test_nodata.tif"
        self.sample.write_image(path=output_path, nodata=-9999)
        reloaded = rs.load_image(output_path)
        self.assertIsNotNone(reloaded.nodata)
        self.assertEqual(reloaded.nodata, -9999.0)

    def test_write_uncompressed(self):
        output_path = Path(tempfile.gettempdir()) / "test_uncompressed.tif"
        self.sample.write_image(path=output_path, compression=False)
        reloaded = rs.load_image(output_path)
        self.assertTrue(np.all(self.sample.array == reloaded.array))

    def test_write_with_extra_metadata(self):
        output_path = Path(tempfile.gettempdir()) / "test_metadata.tif"
        self.sample.write_image(path=output_path, metadata={"SENSOR": "TEST"})
        reloaded = rs.load_image(output_path)
        self.assertEqual(reloaded.metadata.get("SENSOR"), "TEST")

    def test_write_with_nodata_zero(self):
        # nodata=0 must be written, not silently skipped (was broken when `if nodata:`)
        output_path = Path(tempfile.gettempdir()) / "test_nodata_zero.tif"
        self.sample.write_image(path=output_path, nodata=0)
        reloaded = rs.load_image(output_path)
        self.assertIsNotNone(reloaded.nodata)
        self.assertEqual(reloaded.nodata, 0.0)


class TestImageCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_copy_is_independent(self):
        copied = self.sample.copy()
        copied.array[0, 0, 0] = 9999
        self.assertNotEqual(self.sample.array[0, 0, 0], 9999)

    def test_copy_transformation_is_independent(self):
        copied = self.sample.copy()
        copied.transformation[0] = 99999
        self.assertNotEqual(self.sample.transformation[0], 99999)

    def test_copy_preserves_properties(self):
        copied = self.sample.copy()
        self.assertEqual(copied.epsg_code, self.sample.epsg_code)
        self.assertEqual(copied.projection, self.sample.projection)
        self.assertListEqual(copied.transformation, self.sample.transformation)
        np.testing.assert_array_equal(copied.array, self.sample.array)

    def test_copy_preserves_nodata(self):
        # nodata and dtype were added to copy() — verify they are carried over
        img = self.sample.copy()
        img.nodata = -9999
        copied = img.copy()
        self.assertEqual(copied.nodata, -9999)


class TestExtents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_extents_returns_four_values(self):
        bbox = self.sample.extents()
        self.assertEqual(len(bbox), 4)

    def test_extents_xmin_lt_xmax(self):
        xmin, ymin, xmax, ymax = self.sample.extents()
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_extents_transform_to_wgs84(self):
        bbox_wgs = self.sample.extents(dst_epsg=4326)
        xmin, ymin, xmax, ymax = bbox_wgs
        self.assertGreater(xmin, -180)
        self.assertLess(xmax, 180)
        self.assertGreater(ymin, -90)
        self.assertLess(ymax, 90)
        # UTM coordinates are much larger than lon/lat
        bbox_utm = self.sample.extents()
        self.assertNotAlmostEqual(bbox_utm[0], bbox_wgs[0], places=1)


class TestResample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = rs.load_image(SAMPLE_TIF)
        # Use a small crop so resampling is fast
        cls.sample = full.clip(image_extents=[0, 0, 200, 200])

    def test_resample_by_shape(self):
        resampled = self.sample.resample(out_shape=(50, 50))
        self.assertEqual(resampled.array.shape[0], 50)
        self.assertEqual(resampled.array.shape[1], 50)
        self.assertEqual(resampled.array.shape[2], self.sample.array.shape[2])

    def test_resample_by_pixel_size(self):
        psx = self.sample.transformation[1]
        resampled = self.sample.resample(out_pix=psx * 2)
        self.assertLess(resampled.array.shape[0], self.sample.array.shape[0])
        self.assertLess(resampled.array.shape[1], self.sample.array.shape[1])

    def test_resample_inplace_returns_none(self):
        img = copy.deepcopy(self.sample)
        result = img.resample(out_shape=(50, 50), inplace=True)
        self.assertIsNone(result)
        self.assertEqual(img.array.shape[0], 50)

    def test_resample_returns_new_image(self):
        resampled = self.sample.resample(out_shape=(50, 50), inplace=False)
        self.assertIsInstance(resampled, rs.Image)
        self.assertEqual(self.sample.array.shape[0], 200)  # original unchanged

    def test_resample_updates_transformation(self):
        resampled = self.sample.resample(out_shape=(50, 50))
        # Pixel size should have changed
        self.assertNotAlmostEqual(
            resampled.transformation[1], self.sample.transformation[1], places=5
        )

    def test_resample_near_interpolation(self):
        resampled = self.sample.resample(out_shape=(50, 50), interpolation="near")
        self.assertEqual(resampled.array.shape[:2], (50, 50))

    def test_resample_out_pix_single_element_list(self):
        # A length-1 list for out_pix should be doubled to [psx, psx]
        psx = self.sample.transformation[1]
        resampled = self.sample.resample(out_pix=[psx * 2])
        self.assertLess(resampled.array.shape[0], self.sample.array.shape[0])


class TestReproject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = rs.load_image(SAMPLE_TIF)
        cls.sample = full.clip(image_extents=[0, 0, 100, 100])

    def test_reproject_changes_epsg(self):
        reprojected = self.sample.reproject(target_epsg=4326)
        self.assertEqual(int(reprojected.epsg_code), 4326)

    def test_reproject_inplace_returns_none(self):
        img = copy.deepcopy(self.sample)
        result = img.reproject(target_epsg=4326, inplace=True)
        self.assertIsNone(result)
        self.assertEqual(int(img.epsg_code), 4326)

    def test_reproject_returns_new_image(self):
        reprojected = self.sample.reproject(target_epsg=4326, inplace=False)
        self.assertIsInstance(reprojected, rs.Image)
        self.assertEqual(int(self.sample.epsg_code), 32639)  # original unchanged

    def test_reproject_same_epsg(self):
        orig_epsg = int(self.sample.epsg_code)
        reprojected = self.sample.reproject(target_epsg=orig_epsg)
        self.assertEqual(int(reprojected.epsg_code), orig_epsg)


class TestTo8Bit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_output_dtype_is_uint8(self):
        result = self.sample.to_8bit()
        self.assertEqual(result.array.dtype, np.uint8)

    def test_output_values_in_range(self):
        result = self.sample.to_8bit()
        self.assertGreaterEqual(int(result.array.min()), 0)
        self.assertLessEqual(int(result.array.max()), 255)

    def test_to_8bit_inplace_returns_none(self):
        img = copy.deepcopy(self.sample)
        result = img.to_8bit(inplace=True)
        self.assertIsNone(result)
        self.assertEqual(img.array.dtype, np.uint8)

    def test_to_8bit_single_band(self):
        arr = np.random.randint(0, 65535, (50, 50), dtype=np.uint16)
        img = rs.Image(array=arr, transformation=[0, 1, 0, 0, 0, -1])
        result = img.to_8bit()
        self.assertEqual(result.array.dtype, np.uint8)
        self.assertGreaterEqual(int(result.array.min()), 0)
        self.assertLessEqual(int(result.array.max()), 255)

    def test_to_8bit_preserves_shape(self):
        result = self.sample.to_8bit()
        self.assertTupleEqual(result.array.shape, self.sample.array.shape)


class TestToGrayscale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_to_grayscale_reduces_to_2d(self):
        gray = self.sample.to_grayscale()
        self.assertEqual(gray.array.ndim, 2)

    def test_to_grayscale_preserves_spatial_dims(self):
        gray = self.sample.to_grayscale()
        self.assertEqual(gray.array.shape[0], self.sample.array.shape[0])
        self.assertEqual(gray.array.shape[1], self.sample.array.shape[1])

    def test_to_grayscale_already_2d_returns_array(self):
        arr = np.random.randint(0, 256, (50, 50))
        img = rs.Image(array=arr)
        result = img.to_grayscale()
        # Returns the array directly (not an Image) when already 2D
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)

    def test_to_grayscale_inplace_returns_none(self):
        img = copy.deepcopy(self.sample)
        result = img.to_grayscale(inplace=True)
        self.assertIsNone(result)
        self.assertEqual(img.array.ndim, 2)


class TestClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)

    def test_clip_by_image_extents_shape(self):
        clipped = self.sample.clip(image_extents=[0, 0, 100, 100])
        self.assertTupleEqual(clipped.array.shape, (100, 100, 4))

    def test_clip_by_bbox(self):
        xmin, ymin, xmax, ymax = self.sample.extents()
        xmid = xmin + (xmax - xmin) / 2
        ymid = ymin + (ymax - ymin) / 2
        clipped = self.sample.clip(bbox=(xmin, ymid, xmid, ymax))
        self.assertLess(clipped.array.shape[0], self.sample.array.shape[0])
        self.assertLess(clipped.array.shape[1], self.sample.array.shape[1])

    def test_clip_inplace_returns_none(self):
        img = copy.deepcopy(self.sample)
        result = img.clip(image_extents=[0, 0, 100, 100], inplace=True)
        self.assertIsNone(result)
        self.assertTupleEqual(img.array.shape, (100, 100, 4))

    def test_clip_no_args_returns_same_shape(self):
        result = self.sample.clip()
        self.assertTupleEqual(result.array.shape, self.sample.array.shape)

    def test_clip_image_extents_overrides_bbox(self):
        xmin, ymin, xmax, ymax = self.sample.extents()
        clipped = self.sample.clip(
            image_extents=[0, 0, 50, 50],
            bbox=(xmin, ymin, xmax, ymax),
        )
        # image_extents should take precedence
        self.assertTupleEqual(clipped.array.shape, (50, 50, 4))


class TestFindDtype(unittest.TestCase):
    def test_uint8_positive_small(self):
        arr = np.array([0, 100, 200])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "uint8")
        self.assertEqual(dtype_np, np.uint8)

    def test_int8_negative_small(self):
        arr = np.array([-100, 0, 100])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "int8")
        self.assertEqual(dtype_np, np.int8)

    def test_uint16_positive_large(self):
        arr = np.array([0, 1000, 60000])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "uint16")
        self.assertEqual(dtype_np, np.uint16)

    def test_int16_negative_large(self):
        arr = np.array([-1000, 0, 1000])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "int16")
        self.assertEqual(dtype_np, np.int16)

    def test_float32_from_float_values(self):
        arr = np.array([0.1, 0.5, 1.5])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "float32")
        self.assertEqual(dtype_np, np.float32)

    def test_float32_from_very_large_int(self):
        arr = np.array([0, 100000])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "float32")
        self.assertEqual(dtype_np, np.float32)

    def test_boundary_uint8_max(self):
        arr = np.array([0, 255])
        dtype_s, _ = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "uint8")

    def test_boundary_uint16_min(self):
        arr = np.array([0, 256])
        dtype_s, _ = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "uint16")

    def test_empty_array_returns_float32(self):
        arr = np.array([])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "float32")
        self.assertEqual(dtype_np, np.float32)

    def test_fully_masked_array_returns_float32(self):
        # All values masked → compressed() is empty → should return float32
        arr = np.ma.MaskedArray([1, 2, 3], mask=[True, True, True])
        dtype_s, dtype_np = utils.find_dtype(arr)
        self.assertEqual(dtype_s, "float32")
        self.assertEqual(dtype_np, np.float32)

    def test_masked_array_ignores_masked_values(self):
        # Only unmasked values should influence the dtype inference
        arr = np.ma.MaskedArray([0, 100, 99999], mask=[False, False, True])
        dtype_s, _ = utils.find_dtype(arr)
        # max of valid values is 100 < 256, min >= 0 → uint8
        self.assertEqual(dtype_s, "uint8")


class TestSwf(unittest.TestCase):
    def test_swf_2d_mean_preserves_shape(self):
        arr = np.random.rand(30, 30).astype(np.float32)
        result = utils.swf(arr, ksize=3, filter_op="mean")
        self.assertTupleEqual(result.shape, arr.shape)

    def test_swf_2d_median_preserves_shape(self):
        arr = np.random.rand(30, 30).astype(np.float32)
        result = utils.swf(arr, ksize=3, filter_op="median")
        self.assertTupleEqual(result.shape, arr.shape)

    def test_swf_3d_mean_preserves_shape(self):
        arr = np.random.rand(20, 20, 3).astype(np.float32)
        result = utils.swf(arr, ksize=3, filter_op="mean")
        self.assertTupleEqual(result.shape, arr.shape)

    def test_swf_3d_median_preserves_shape(self):
        arr = np.random.rand(20, 20, 3).astype(np.float32)
        result = utils.swf(arr, ksize=3, filter_op="median")
        self.assertTupleEqual(result.shape, arr.shape)

    def test_swf_even_ksize_adjusted_to_odd(self):
        # ksize=4 is even; should be bumped to 5, result still has same shape
        arr = np.random.rand(20, 20).astype(np.float32)
        result = utils.swf(arr, ksize=4, filter_op="mean")
        self.assertTupleEqual(result.shape, arr.shape)

    def test_swf_callable_filter(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        result = utils.swf(arr, ksize=3, filter_op=np.mean)
        self.assertTupleEqual(result.shape, arr.shape)


class TestGetSlidingWin(unittest.TestCase):
    def test_2d_with_padding(self):
        arr = np.random.randint(0, 256, (100, 100))
        result = utils.get_sliding_win(arr, ksize=3, step_x=1, step_y=1, pad=True)
        self.assertTupleEqual(result.shape, (102, 102, 3, 3))

    def test_3d_with_padding(self):
        arr = np.random.randint(0, 256, (100, 100, 3))
        result = utils.get_sliding_win(arr, ksize=3, step_x=1, step_y=1, pad=True)
        self.assertTupleEqual(result.shape, (102, 102, 3, 3, 3))

    def test_2d_no_padding(self):
        arr = np.random.randint(0, 256, (10, 10))
        # (10 - 3) / 1 + 1 = 8
        result = utils.get_sliding_win(arr, ksize=3, step_x=1, step_y=1, pad=False)
        self.assertTupleEqual(result.shape, (8, 8, 3, 3))

    def test_custom_step(self):
        arr = np.random.randint(0, 256, (10, 10))
        # floor((10 - 3) / 2) + 1 = 4
        result = utils.get_sliding_win(arr, ksize=3, step_x=2, step_y=2, pad=False)
        self.assertTupleEqual(result.shape, (4, 4, 3, 3))

    def test_invalid_array_shape_raises(self):
        arr = np.random.randint(0, 256, (10, 10, 3, 2))
        with self.assertRaises(ValueError):
            utils.get_sliding_win(arr, ksize=3, pad=False)

    def test_2d_window_values_match_input(self):
        # Each window[i, j] must equal arr[i:i+ksize, j:j+ksize]
        arr = np.random.randint(0, 256, (8, 8))
        ksize = 3
        windows = utils.get_sliding_win(arr, ksize=ksize, step_x=1, step_y=1, pad=False)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                np.testing.assert_array_equal(
                    windows[i, j], arr[i : i + ksize, j : j + ksize]
                )

    def test_image_sliding_win_returns_object_array_of_images(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 20, 20])
        windows = utils.get_sliding_win(img, ksize=3, step_x=2, step_y=2, pad=False)
        # (20 - 3) // 2 + 1 = 9
        self.assertEqual(windows.shape, (9, 9))
        self.assertIsInstance(windows[0, 0], rs.Image)

    def test_image_sliding_win_tile_shape(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 20, 20])
        windows = utils.get_sliding_win(img, ksize=5, step_x=1, step_y=1, pad=False)
        self.assertTupleEqual(windows[0, 0].array.shape, (5, 5, 4))

    def test_image_sliding_win_geotransform_no_pad(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 20, 20])
        psx = img.transformation[1]

        windows = utils.get_sliding_win(img, ksize=3, step_x=2, step_y=2, pad=False)
        # window (0,0) starts at origin (no padding, no shift)
        self.assertAlmostEqual(windows[0, 0].transformation[0], img.transformation[0])
        self.assertAlmostEqual(windows[0, 0].transformation[3], img.transformation[3])
        # window (0,1) is shifted by step_x=2 pixels
        self.assertAlmostEqual(
            windows[0, 1].transformation[0], img.transformation[0] + 2 * psx
        )

    def test_image_sliding_win_geotransform_with_pad(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 20, 20])
        ksize = 3
        radius = ksize // 2  # = 1
        psx = img.transformation[1]

        windows = utils.get_sliding_win(img, ksize=ksize, step_x=1, step_y=1, pad=True)
        # window (0,0) with padding: origin is shifted by -radius pixels
        self.assertAlmostEqual(
            windows[0, 0].transformation[0], img.transformation[0] - radius * psx
        )
        # window (1,1) corresponds to pixel (1,1) of the original image → same origin
        self.assertAlmostEqual(windows[1, 1].transformation[0], img.transformation[0])

    def test_3d_window_values_match_input(self):
        arr = np.random.randint(0, 256, (8, 8, 3))
        ksize = 3
        windows = utils.get_sliding_win(arr, ksize=ksize, step_x=1, step_y=1, pad=False)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                np.testing.assert_array_equal(
                    windows[i, j], arr[i : i + ksize, j : j + ksize, :]
                )


class TestGetTiles(unittest.TestCase):
    def test_2d_ksize(self):
        arr = np.random.randint(0, 256, (100, 100))
        result = utils.get_tiles(arr, ksize=3)
        self.assertTupleEqual(result.shape, (34, 34, 3, 3))

    def test_3d_ksize(self):
        arr = np.random.randint(0, 256, (100, 100, 3))
        result = utils.get_tiles(arr, ksize=3)
        self.assertTupleEqual(result.shape, (34, 34, 3, 3, 3))

    def test_nblocks_scalar(self):
        arr = np.random.randint(0, 256, (100, 100))
        # 10 blocks → ksize = 10 → 10x10 tiles of size 10x10
        result = utils.get_tiles(arr, nblocks=10, pad=False)
        self.assertEqual(result.shape[0], 10)
        self.assertEqual(result.shape[1], 10)
        self.assertEqual(result.shape[2], 10)
        self.assertEqual(result.shape[3], 10)

    def test_nblocks_tuple(self):
        arr = np.random.randint(0, 256, (100, 100))
        # nblocks=(5, 10) → ksize_x=100//5=20, ksize_y=100//10=10
        result = utils.get_tiles(arr, nblocks=(5, 10), pad=False)
        self.assertEqual(result.shape[0], 10)
        self.assertEqual(result.shape[1], 5)

    def test_tuple_ksize(self):
        arr = np.random.randint(0, 256, (100, 100))
        # ksize=(5, 10): ksize_x=5, ksize_y=10, step_x=5, step_y=10
        # out_shape = (100//10, 100//5, 10, 5) = (10, 20, 10, 5)
        result = utils.get_tiles(arr, ksize=(5, 10), pad=False)
        self.assertTupleEqual(result.shape, (10, 20, 10, 5))

    def test_no_ksize_or_nblocks_raises(self):
        arr = np.random.randint(0, 256, (100, 100))
        with self.assertRaises(RuntimeError):
            utils.get_tiles(arr)

    def test_invalid_array_shape_raises(self):
        # 4D array is not supported
        arr = np.random.randint(0, 256, (10, 10, 3, 2))
        with self.assertRaises(ValueError):
            utils.get_tiles(arr, ksize=3, pad=False)

    def test_2d_round_trip(self):
        # Use a shape exactly divisible by ksize so no padding is needed
        arr = np.random.randint(0, 256, (30, 30))
        tiles = utils.get_tiles(arr, ksize=5, pad=False)  # (6, 6, 5, 5)
        ny, nx, ky, kx = tiles.shape
        reconstructed = tiles.transpose(0, 2, 1, 3).reshape(ny * ky, nx * kx)
        np.testing.assert_array_equal(reconstructed, arr)

    def test_3d_round_trip(self):
        arr = np.random.randint(0, 256, (30, 30, 4))
        tiles = utils.get_tiles(arr, ksize=5, pad=False)  # (6, 6, 5, 5, 4)
        ny, nx, ky, kx, C = tiles.shape
        reconstructed = tiles.transpose(0, 2, 1, 3, 4).reshape(ny * ky, nx * kx, C)
        np.testing.assert_array_equal(reconstructed, arr)

    def test_image_tiles_returns_object_array_of_images(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 100, 100])
        tiles = utils.get_tiles(img, ksize=25, pad=False)
        self.assertEqual(tiles.shape, (4, 4))
        self.assertIsInstance(tiles[0, 0], rs.Image)

    def test_image_tiles_correct_shape(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 100, 100])
        tiles = utils.get_tiles(img, ksize=25, pad=False)
        self.assertTupleEqual(tiles[0, 0].array.shape, (25, 25, 4))

    def test_image_tiles_geotransform_origin(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 100, 100])
        psx = img.transformation[1]
        psy = img.transformation[5]
        tiles = utils.get_tiles(img, ksize=25, pad=False)
        # tile (0,0) origin matches the image origin
        self.assertAlmostEqual(tiles[0, 0].transformation[0], img.transformation[0])
        self.assertAlmostEqual(tiles[0, 0].transformation[3], img.transformation[3])
        # tile (0,1) x-origin is shifted by 25 pixels
        self.assertAlmostEqual(
            tiles[0, 1].transformation[0], img.transformation[0] + 25 * psx
        )
        # tile (1,0) y-origin is shifted by 25 pixels
        self.assertAlmostEqual(
            tiles[1, 0].transformation[3], img.transformation[3] + 25 * psy
        )

    def test_image_tiles_consistent_projection(self):
        img = rs.load_image(SAMPLE_TIF).clip(image_extents=[0, 0, 50, 50])
        tiles = utils.get_tiles(img, ksize=25, pad=False)
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                self.assertEqual(tiles[i, j].projection, img.projection)
                self.assertEqual(tiles[i, j].epsg_code, img.epsg_code)

    def test_divisible_array_no_overpadding(self):
        # Array exactly divisible by ksize: pad=True must not add extra rows/cols
        arr = np.random.randint(0, 256, (9, 9))
        result = utils.get_tiles(arr, ksize=3, pad=True)
        self.assertTupleEqual(result.shape, (3, 3, 3, 3))

    def test_3d_divisible_array_no_overpadding(self):
        arr = np.random.randint(0, 256, (12, 12, 4))
        result = utils.get_tiles(arr, ksize=4, pad=True)
        self.assertTupleEqual(result.shape, (3, 3, 4, 4, 4))


class TestLoadFromZip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_zip = Path(tempfile.gettempdir()) / "test_resens.zip"
        with zipfile.ZipFile(cls.tmp_zip, "w") as zf:
            zf.write(SAMPLE_TIF, arcname="B1_sample.tif")
            zf.write(SAMPLE_TIF, arcname="B2_sample.tif")

        cls.tmp_tar = Path(tempfile.gettempdir()) / "test_resens.tar"
        with tarfile.open(cls.tmp_tar, "w") as tf:
            tf.add(SAMPLE_TIF, arcname="B1_sample.tif")
            tf.add(SAMPLE_TIF, arcname="B2_sample.tif")

        cls.tmp_tar_gz = Path(tempfile.gettempdir()) / "test_resens.tar.gz"
        with tarfile.open(cls.tmp_tar_gz, "w:gz") as tf:
            tf.add(SAMPLE_TIF, arcname="B1_sample.tif")
            tf.add(SAMPLE_TIF, arcname="B2_sample.tif")

    def test_load_multiple_patterns(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_zip,
            patterns=["B1", "B2"],
            extension=".tif",
        )
        self.assertIn("B1", result)
        self.assertIn("B2", result)
        self.assertIsInstance(result["B1"], rs.Image)
        self.assertIsInstance(result["B2"], rs.Image)

    def test_load_single_pattern_as_string(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_zip,
            patterns="B1",
            extension=".tif",
        )
        self.assertIn("B1", result)
        self.assertIsInstance(result["B1"], rs.Image)

    def test_load_with_group_filter(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_zip,
            patterns=["B1", "B2"],
            extension=".tif",
            group="sample",
        )
        self.assertEqual(len(result), 2)

    def test_load_with_group_no_match(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_zip,
            patterns=["B1", "B2"],
            extension=".tif",
            group="nonexistent",
        )
        self.assertEqual(len(result), 0)

    def test_missing_zip_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            rs.load_from_zip(
                zipf_path=Path("/nonexistent/path.zip"),
                patterns=["B1"],
                extension=".tif",
            )

    def test_bad_zip_returns_none(self):
        bad_zip = Path(tempfile.gettempdir()) / "bad.zip"
        bad_zip.write_bytes(b"not a zip file")
        result = rs.load_from_zip(
            zipf_path=bad_zip,
            patterns=["B1"],
            extension=".tif",
        )
        self.assertIsNone(result)

    def test_loaded_image_has_correct_shape(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_zip,
            patterns=["B1"],
            extension=".tif",
        )
        self.assertTupleEqual(result["B1"].array.shape, (1511, 1441, 4))

    def test_load_from_zip_string_path(self):
        # zipf_path passed as str should be converted to Path internally
        result = rs.load_from_zip(
            zipf_path=str(self.tmp_zip),
            patterns=["B1"],
            extension=".tif",
        )
        self.assertIn("B1", result)

    def test_load_from_zip_duplicate_key_skipped(self):
        # Two files matching the same pattern key: the second should be skipped
        dup_zip = Path(tempfile.gettempdir()) / "test_dup.zip"
        with zipfile.ZipFile(dup_zip, "w") as zf:
            zf.write(SAMPLE_TIF, arcname="B1_first.tif")
            zf.write(SAMPLE_TIF, arcname="B1_second.tif")
        result = rs.load_from_zip(
            zipf_path=dup_zip,
            patterns=["B1"],
            extension=".tif",
        )
        # Only one image should be loaded for key "B1" (second skipped)
        self.assertIn("B1", result)
        self.assertEqual(len(result), 1)

    def test_load_from_tar(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_tar,
            patterns=["B1", "B2"],
            extension=".tif",
        )
        self.assertIn("B1", result)
        self.assertIn("B2", result)
        self.assertIsInstance(result["B1"], rs.Image)
        self.assertTupleEqual(result["B1"].array.shape, (1511, 1441, 4))

    def test_load_from_tar_gz(self):
        result = rs.load_from_zip(
            zipf_path=self.tmp_tar_gz,
            patterns=["B1", "B2"],
            extension=".tif",
        )
        self.assertIn("B1", result)
        self.assertIn("B2", result)
        self.assertIsInstance(result["B1"], rs.Image)
        self.assertTupleEqual(result["B1"].array.shape, (1511, 1441, 4))

    def test_load_from_tgz(self):
        tgz = Path(tempfile.gettempdir()) / "test_resens.tgz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(SAMPLE_TIF, arcname="B1_sample.tif")
        result = rs.load_from_zip(zipf_path=tgz, patterns=["B1"], extension=".tif")
        self.assertIn("B1", result)
        self.assertIsInstance(result["B1"], rs.Image)

    def test_load_from_bad_tar_returns_none(self):
        bad_tar = Path(tempfile.gettempdir()) / "bad.tar"
        bad_tar.write_bytes(b"not a tar file")
        result = rs.load_from_zip(
            zipf_path=bad_tar,
            patterns=["B1"],
            extension=".tif",
        )
        self.assertIsNone(result)


class TestShapefileMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = rs.load_image(SAMPLE_TIF)
        xmin, ymin, xmax, ymax = cls.sample.extents()
        # A small square polygon in the center of the image
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        size = (xmax - xmin) / 10
        polygon = sg.box(cx - size, cy - size, cx + size, cy + size)
        cls.gdf = gpd.GeoDataFrame(
            geometry=[polygon],
            crs=f"EPSG:{cls.sample.epsg_code}",
        )

    def test_masking_returns_image(self):
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
        )
        self.assertIsInstance(mask, rs.Image)

    def test_masking_correct_shape(self):
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
        )
        self.assertTupleEqual(mask.array.shape[:2], self.sample.array.shape[:2])

    def test_masking_contains_burned_value(self):
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            burn_value=1,
        )
        self.assertGreater(int(np.sum(mask.array == 1)), 0)

    def test_masking_background_is_zero(self):
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            burn_value=1,
        )
        self.assertGreater(int(np.sum(mask.array == 0)), 0)

    def test_masking_with_dilation_expands_mask(self):
        mask_plain = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
        )
        mask_dilated = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            dilation=True,
            dilation_iters=3,
        )
        self.assertGreaterEqual(
            int(np.sum(mask_dilated.array > 0)),
            int(np.sum(mask_plain.array > 0)),
        )

    def test_masking_custom_burn_value(self):
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            burn_value=5,
        )
        self.assertGreater(int(np.sum(mask.array == 5)), 0)

    def test_masking_gdf_in_different_crs_is_reprojected(self):
        # GDF in EPSG:4326 (lat/lon) must be reprojected to match the raster CRS
        xmin, ymin, xmax, ymax = self.sample.extents(dst_epsg=4326)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        size = (xmax - xmin) / 20
        polygon = sg.box(cx - size, cy - size, cx + size, cy + size)
        gdf_4326 = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        mask = shapefile_masking(
            gdf=gdf_4326,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
        )
        self.assertIsInstance(mask, rs.Image)
        self.assertGreater(int(np.sum(mask.array > 0)), 0)

    def test_masking_with_explicit_outpath_no_suffix(self):
        # mask_outpath without .tif suffix → UUID filename is appended inside the dir
        out_dir = Path(tempfile.gettempdir()) / "resens_mask_test"
        mask = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            mask_outpath=out_dir,
        )
        self.assertIsInstance(mask, rs.Image)
        # The output file should exist inside the directory
        tif_files = list(out_dir.glob("*.tif"))
        self.assertGreater(len(tif_files), 0)

    def test_masking_dilation_without_iters_defaults_to_one(self):
        # dilation=True with no dilation_iters uses the default of 1 iteration
        mask_no_dil = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
        )
        mask_dil = shapefile_masking(
            gdf=self.gdf,
            shape=self.sample.array.shape,
            transformation=self.sample.transformation,
            projection=self.sample.projection,
            dilation=True,
            # dilation_iters intentionally omitted
        )
        self.assertGreaterEqual(
            int(np.sum(mask_dil.array > 0)),
            int(np.sum(mask_no_dil.array > 0)),
        )


if __name__ == "__main__":
    unittest.main()
