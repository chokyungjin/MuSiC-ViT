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

from monai.transforms import RandHistogramShift

TEST_CASES = [
    [
        {"num_control_points": 5, "prob": 0.0},
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
        np.arange(8).reshape((1, 2, 2, 2)),
    ],
    [
        {"num_control_points": 5, "prob": 0.9},
        {"img": np.arange(8).reshape((1, 2, 2, 2)).astype(np.float32)},
        np.array([[[[0.0, 0.57227867], [1.1391707, 1.68990281]], [[2.75833219, 4.34445884], [5.70913743, 7.0]]]]),
    ],
    [
        {"num_control_points": (5, 20), "prob": 0.9},
        {"img": np.arange(8).reshape((1, 2, 2, 2)).astype(np.float32)},
        np.array([[[[0.0, 1.17472492], [2.21553091, 2.88292011]], [[3.98407301, 5.01302123], [6.09275004, 7.0]]]]),
    ],
]


class TestRandHistogramShift(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_histogram_shift(self, input_param, input_data, expected_val):
        g = RandHistogramShift(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
