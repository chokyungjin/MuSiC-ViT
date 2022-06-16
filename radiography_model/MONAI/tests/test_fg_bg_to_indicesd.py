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

from monai.transforms import FgBgToIndicesd

TEST_CASE_1 = [
    {"keys": "label", "image_key": None, "image_threshold": 0.0, "output_shape": None},
    {"label": np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])},
    np.array([1, 2, 3, 5, 6, 7]),
    np.array([0, 4, 8]),
]

TEST_CASE_2 = [
    {"keys": "label", "image_key": "image", "image_threshold": 0.0, "output_shape": None},
    {"label": np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]), "image": np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])},
    np.array([1, 2, 3, 5, 6, 7]),
    np.array([0, 8]),
]

TEST_CASE_3 = [
    {"keys": "label", "image_key": "image", "image_threshold": 1.0, "output_shape": None},
    {"label": np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]), "image": np.array([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]])},
    np.array([1, 2, 3, 5, 6, 7]),
    np.array([0, 8]),
]

TEST_CASE_4 = [
    {"keys": "label", "image_key": "image", "image_threshold": 1.0, "output_shape": None},
    {"label": np.array([[[0, 1, 2], [3, 0, 4], [5, 6, 0]]]), "image": np.array([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]])},
    np.array([1, 2, 3, 5, 6, 7]),
    np.array([0, 8]),
]

TEST_CASE_5 = [
    {"keys": "label", "image_key": "image", "image_threshold": 1.0, "output_shape": [3, 3]},
    {"label": np.array([[[0, 1, 2], [3, 0, 4], [5, 6, 0]]]), "image": np.array([[[3, 3, 3], [3, 1, 3], [3, 3, 3]]])},
    np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]),
    np.array([[0, 0], [2, 2]]),
]


class TestFgBgToIndicesd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_type_shape(self, input_data, data, expected_fg, expected_bg):
        result = FgBgToIndicesd(**input_data)(data)
        np.testing.assert_allclose(result["label_fg_indices"], expected_fg)
        np.testing.assert_allclose(result["label_bg_indices"], expected_bg)


if __name__ == "__main__":
    unittest.main()
