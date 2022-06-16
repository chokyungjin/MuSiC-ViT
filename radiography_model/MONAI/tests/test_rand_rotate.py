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

from monai.transforms import RandRotate
from tests.utils import NumpyImageTestCase2D, NumpyImageTestCase3D


class TestRandRotate2D(NumpyImageTestCase2D):
    @parameterized.expand(
        [
            (np.pi / 2, True, "bilinear", "border", False),
            (np.pi / 4, True, "nearest", "border", False),
            (np.pi, False, "nearest", "zeros", True),
            ((-np.pi / 4, 0), False, "nearest", "zeros", True),
        ]
    )
    def test_correct_results(self, degrees, keep_size, mode, padding_mode, align_corners):
        rotate_fn = RandRotate(
            range_x=degrees,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn(self.imt[0])

        _order = 0 if mode == "nearest" else 1
        if mode == "border":
            _mode = "nearest"
        elif mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        angle = rotate_fn.x
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -np.rad2deg(angle), (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(np.float32)
        good = np.sum(np.isclose(expected, rotated[0], atol=1e-3))
        self.assertLessEqual(np.abs(good - expected.size), 5, "diff at most 5 pixels")


class TestRandRotate3D(NumpyImageTestCase3D):
    @parameterized.expand(
        [
            (np.pi / 2, -np.pi / 6, (0.0, np.pi), False, "bilinear", "border", False, (1, 87, 104, 109)),
            (
                np.pi / 4,
                (-np.pi / 9, np.pi / 4.5),
                (np.pi / 9, np.pi / 6),
                False,
                "nearest",
                "border",
                True,
                (1, 89, 105, 104),
            ),
            (
                0.0,
                (2 * np.pi, 2.06 * np.pi),
                (-np.pi / 180, np.pi / 180),
                True,
                "nearest",
                "zeros",
                True,
                (1, 48, 64, 80),
            ),
            ((-np.pi / 4, 0), 0, 0, False, "nearest", "zeros", False, (1, 48, 77, 90)),
        ]
    )
    def test_correct_results(self, x, y, z, keep_size, mode, padding_mode, align_corners, expected):
        rotate_fn = RandRotate(
            range_x=x,
            range_y=y,
            range_z=z,
            prob=1.0,
            keep_size=keep_size,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        rotate_fn.set_random_state(243)
        rotated = rotate_fn(self.imt[0])
        np.testing.assert_allclose(rotated.shape, expected)


if __name__ == "__main__":
    unittest.main()
