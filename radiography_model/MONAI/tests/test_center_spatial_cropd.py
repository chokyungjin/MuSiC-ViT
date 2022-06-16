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
from parameterized import parameterized

from monai.transforms import CenterSpatialCropd

TEST_CASE_0 = [
    {"keys": "img", "roi_size": [2, -1, -1]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 3, 3),
]

TEST_CASE_1 = [
    {"keys": "img", "roi_size": [2, 2, 2]},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 2, 2, 2),
]

TEST_CASE_2 = [
    {"keys": "img", "roi_size": [2, 2]},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])},
    np.array([[[1, 2], [2, 3]]]),
]


class TestCenterSpatialCropd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    def test_shape(self, input_param, input_data, expected_shape):
        result = CenterSpatialCropd(**input_param)(input_data)
        self.assertTupleEqual(result["img"].shape, expected_shape)

    @parameterized.expand([TEST_CASE_2])
    def test_value(self, input_param, input_data, expected_value):
        result = CenterSpatialCropd(**input_param)(input_data)
        np.testing.assert_allclose(result["img"], expected_value)


if __name__ == "__main__":
    unittest.main()
