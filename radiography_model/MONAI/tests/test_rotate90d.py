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

from monai.transforms import Rotate90d
from tests.utils import NumpyImageTestCase2D


class TestRotate90d(NumpyImageTestCase2D):
    def test_rotate90_default(self):
        key = "test"
        rotate = Rotate90d(keys=key)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 1, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_k(self):
        key = None
        rotate = Rotate90d(keys=key, k=2)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 2, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_spatial_axes(self):
        key = "test"
        rotate = Rotate90d(keys=key, spatial_axes=(0, 1))
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 1, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_prob_k_spatial_axes(self):
        key = "test"
        rotate = Rotate90d(keys=key, k=2, spatial_axes=(0, 1))
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 2, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_no_key(self):
        key = "unknown"
        rotate = Rotate90d(keys=key)
        with self.assertRaisesRegex(KeyError, ""):
            rotate({"test": self.imt[0]})


if __name__ == "__main__":
    unittest.main()
