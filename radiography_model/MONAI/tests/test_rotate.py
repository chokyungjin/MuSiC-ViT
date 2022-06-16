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

import unittest

import numpy as np
import scipy.ndimage
from parameterized import parameterized

from monai.transforms import Rotate
from tests.utils import NumpyImageTestCase2D, NumpyImageTestCase3D

TEST_CASES_2D = [
    (np.pi / 6, False, "bilinear", "border", False),
    (np.pi / 4, True, "bilinear", "border", False),
    (-np.pi / 4.5, True, "nearest", "reflection", False),
    (np.pi, False, "nearest", "zeros", False),
    (-np.pi / 2, False, "bilinear", "zeros", True),
]

TEST_CASES_3D = [
    (-np.pi / 2, True, "nearest", "border", False),
    (np.pi / 4, True, "bilinear", "border", False),
    (-np.pi / 4.5, True, "nearest", "reflection", False),
    (np.pi, False, "nearest", "zeros", False),
    (-np.pi / 2, False, "bilinear", "zeros", False),
]

TEST_CASES_SHAPE_3D = [
    ([-np.pi / 2, 1.0, 2.0], "nearest", "border", False),
    ([np.pi / 4, 0, 0], "bilinear", "border", False),
    ([-np.pi / 4.5, -20, 20], "nearest", "reflection", False),
]


class TestRotate2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotate(angle, keep_size, mode, padding_mode, align_corners)
        rotated = rotate_fn(self.imt[0])
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"

        expected = []
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel,
                    -np.rad2deg(angle),
                    (0, 1),
                    not keep_size,
                    order=_order,
                    mode=_mode,
                    prefilter=False,
                )
            )
        expected = np.stack(expected).astype(np.float32)
        good = np.sum(np.isclose(expected, rotated, atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 pixels")


class TestRotate3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotate([angle, 0, 0], keep_size, mode, padding_mode, align_corners)
        rotated = rotate_fn(self.imt[0])
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated.shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"

        expected = []
        for channel in self.imt[0]:
            expected.append(
                scipy.ndimage.rotate(
                    channel,
                    -np.rad2deg(angle),
                    (1, 2),
                    not keep_size,
                    order=_order,
                    mode=_mode,
                    prefilter=False,
                )
            )
        expected = np.stack(expected).astype(np.float32)
        n_good = np.sum(np.isclose(expected, rotated, atol=1e-3))
        self.assertLessEqual(expected.size - n_good, 5, "diff at most 5 pixels")

    @parameterized.expand(TEST_CASES_SHAPE_3D)
    def test_correct_shape(self, angle, mode, padding_mode, align_corners):
        rotate_fn = Rotate(angle, True, align_corners=align_corners)
        rotated = rotate_fn(self.imt[0], mode=mode, padding_mode=padding_mode)
        np.testing.assert_allclose(self.imt[0].shape, rotated.shape)

    def test_ill_case(self):
        rotate_fn = Rotate(10, True)
        with self.assertRaises(ValueError):  # wrong shape
            rotate_fn(self.imt)

        rotate_fn = Rotate(10, keep_size=False)
        with self.assertRaises(ValueError):  # wrong mode
            rotate_fn(self.imt[0], mode="trilinear")


if __name__ == "__main__":
    unittest.main()
