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

from monai.transforms import RandSpatialCropSamples

TEST_CASE_1 = [
    {"roi_size": [3, 3, 3], "num_samples": 4, "random_center": True, "random_size": False},
    np.arange(192).reshape(3, 4, 4, 4),
    [(3, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3)],
    np.array(
        [
            [
                [[21, 22, 23], [25, 26, 27], [29, 30, 31]],
                [[37, 38, 39], [41, 42, 43], [45, 46, 47]],
                [[53, 54, 55], [57, 58, 59], [61, 62, 63]],
            ],
            [
                [[85, 86, 87], [89, 90, 91], [93, 94, 95]],
                [[101, 102, 103], [105, 106, 107], [109, 110, 111]],
                [[117, 118, 119], [121, 122, 123], [125, 126, 127]],
            ],
            [
                [[149, 150, 151], [153, 154, 155], [157, 158, 159]],
                [[165, 166, 167], [169, 170, 171], [173, 174, 175]],
                [[181, 182, 183], [185, 186, 187], [189, 190, 191]],
            ],
        ]
    ),
]

TEST_CASE_2 = [
    {"roi_size": [3, 3, 3], "num_samples": 8, "random_center": False, "random_size": True},
    np.arange(192).reshape(3, 4, 4, 4),
    [(3, 4, 4, 3), (3, 4, 3, 3), (3, 3, 4, 4), (3, 4, 4, 4), (3, 3, 3, 4), (3, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3)],
    np.array(
        [
            [
                [[21, 22, 23], [25, 26, 27], [29, 30, 31]],
                [[37, 38, 39], [41, 42, 43], [45, 46, 47]],
                [[53, 54, 55], [57, 58, 59], [61, 62, 63]],
            ],
            [
                [[85, 86, 87], [89, 90, 91], [93, 94, 95]],
                [[101, 102, 103], [105, 106, 107], [109, 110, 111]],
                [[117, 118, 119], [121, 122, 123], [125, 126, 127]],
            ],
            [
                [[149, 150, 151], [153, 154, 155], [157, 158, 159]],
                [[165, 166, 167], [169, 170, 171], [173, 174, 175]],
                [[181, 182, 183], [185, 186, 187], [189, 190, 191]],
            ],
        ]
    ),
]


class TestRandSpatialCropSamples(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape, expected_last_item):
        xform = RandSpatialCropSamples(**input_param)
        xform.set_random_state(1234)
        result = xform(input_data)

        np.testing.assert_equal(len(result), input_param["num_samples"])
        for item, expected in zip(result, expected_shape):
            self.assertTupleEqual(item.shape, expected)
        np.testing.assert_allclose(result[-1], expected_last_item)


if __name__ == "__main__":
    unittest.main()
