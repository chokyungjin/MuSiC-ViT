# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import itk
import nibabel as nib
import numpy as np
from parameterized import parameterized
from PIL import Image

from monai.data import ITKReader, NibabelReader
from monai.transforms import LoadImage

TEST_CASE_1 = [{"image_only": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_2 = [{"image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_3 = [
    {"image_only": True},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_4 = [
    {"image_only": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_5 = [
    {"reader": NibabelReader(mmap=False), "image_only": False},
    ["test_image.nii.gz"],
    (128, 128, 128),
]

TEST_CASE_6 = [{"reader": ITKReader(), "image_only": True}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_7 = [{"reader": ITKReader(), "image_only": False}, ["test_image.nii.gz"], (128, 128, 128)]

TEST_CASE_8 = [
    {"reader": ITKReader(), "image_only": True},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_9 = [
    {"reader": ITKReader(), "image_only": False},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    (3, 128, 128, 128),
]

TEST_CASE_10 = [
    {"image_only": False, "reader": ITKReader(pixel_type=itk.UC)},
    "tests/testing_data/CT_DICOM",
    (4, 16, 16),
]

TEST_CASE_11 = [
    {"image_only": False, "reader": "ITKReader", "pixel_type": itk.UC},
    "tests/testing_data/CT_DICOM",
    (4, 16, 16),
]


class TestLoadImage(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_nibabel_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadImage(**input_param)(filenames)

            if isinstance(result, tuple):
                result, header = result
                self.assertTrue("affine" in header)
                self.assertEqual(header["filename_or_obj"], os.path.join(tempdir, "test_image.nii.gz"))
                np.testing.assert_allclose(header["affine"], np.eye(4))
                np.testing.assert_allclose(header["original_affine"], np.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_6, TEST_CASE_7, TEST_CASE_8, TEST_CASE_9])
    def test_itk_reader(self, input_param, filenames, expected_shape):
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                itk_np_view = itk.image_view_from_array(test_image)
                itk.imwrite(itk_np_view, filenames[i])
            result = LoadImage(**input_param)(filenames)

            if isinstance(result, tuple):
                result, header = result
                self.assertTrue("affine" in header)
                self.assertEqual(header["filename_or_obj"], os.path.join(tempdir, "test_image.nii.gz"))
                np.testing.assert_allclose(header["affine"], np.eye(4))
                np.testing.assert_allclose(header["original_affine"], np.eye(4))
            self.assertTupleEqual(result.shape, expected_shape)

    @parameterized.expand([TEST_CASE_10, TEST_CASE_11])
    def test_itk_dicom_series_reader(self, input_param, filenames, expected_shape):
        result, header = LoadImage(**input_param)(filenames)
        self.assertTrue("affine" in header)
        self.assertEqual(header["filename_or_obj"], filenames)
        np.testing.assert_allclose(
            header["affine"],
            np.array(
                [
                    [0.488281, 0.0, 0.0, -125.0],
                    [0.0, 0.488281, 0.0, -128.100006],
                    [0.0, 0.0, 68.33333333, -99.480003],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        self.assertTupleEqual(result.shape, expected_shape)
        self.assertTupleEqual(tuple(header["spatial_shape"]), expected_shape)

    def test_itk_reader_multichannel(self):
        test_image = np.random.randint(0, 256, size=(256, 256, 3)).astype("uint8")
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            itk_np_view = itk.image_view_from_array(test_image, is_vector=True)
            itk.imwrite(itk_np_view, filename)
            result, header = LoadImage(reader=ITKReader())(filename)

            self.assertTupleEqual(tuple(header["spatial_shape"]), (256, 256))
            np.testing.assert_allclose(result[0, :, :], test_image[:, :, 0])
            np.testing.assert_allclose(result[1, :, :], test_image[:, :, 1])
            np.testing.assert_allclose(result[2, :, :], test_image[:, :, 2])

    def test_load_png(self):
        spatial_size = (256, 256)
        test_image = np.random.randint(0, 256, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result, header = LoadImage(image_only=False)(filename)
            self.assertTupleEqual(tuple(header["spatial_shape"]), spatial_size)
            self.assertTupleEqual(result.shape, spatial_size)
            np.testing.assert_allclose(result, test_image)

    def test_register(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=False)
            loader.register(ITKReader())
            result, header = loader(filename)
            self.assertTupleEqual(tuple(header["spatial_shape"]), spatial_size)
            self.assertTupleEqual(result.shape, spatial_size)

    def test_kwargs(self):
        spatial_size = (32, 64, 128)
        expected_shape = (128, 64, 32)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImage(image_only=False)
            reader = ITKReader(fallback_only=False)
            loader.register(reader)
            result, header = loader(filename)

            reader = ITKReader()
            img = reader.read(filename, fallback_only=False)
            result_raw, header_raw = reader.get_data(img)
            np.testing.assert_allclose(header["spatial_shape"], header_raw["spatial_shape"])
            self.assertTupleEqual(result.shape, result_raw.shape)


if __name__ == "__main__":
    unittest.main()
