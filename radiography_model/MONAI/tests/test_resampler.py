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
import torch
from parameterized import parameterized

from monai.transforms import Resample
from monai.transforms.utils import create_grid

TEST_CASES = [
    [
        dict(padding_mode="zeros", as_tensor_output=False, device=None),
        {"grid": create_grid((2, 2)), "img": np.arange(4).reshape((1, 2, 2))},
        np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    ],
    [
        dict(padding_mode="zeros", as_tensor_output=False, device=None),
        {"grid": create_grid((4, 4)), "img": np.arange(4).reshape((1, 2, 2))},
        np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
    ],
    [
        dict(padding_mode="border", as_tensor_output=False, device=None),
        {"grid": create_grid((4, 4)), "img": np.arange(4).reshape((1, 2, 2))},
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3, 3.0], [2.0, 2.0, 3.0, 3.0]]]),
    ],
    [
        dict(padding_mode="reflection", as_tensor_output=False, device=None),
        {"grid": create_grid((4, 4)), "img": np.arange(4).reshape((1, 2, 2)), "mode": "nearest"},
        np.array([[[3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0], [3.0, 2.0, 3.0, 2.0], [1.0, 0.0, 1.0, 0.0]]]),
    ],
    [
        dict(padding_mode="zeros", as_tensor_output=False, device=None),
        {"grid": create_grid((4, 4, 4)), "img": np.arange(8).reshape((1, 2, 2, 2)), "mode": "bilinear"},
        np.array(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 5.0, 0.0], [0.0, 6.0, 7.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        ),
    ],
    [
        dict(padding_mode="border", as_tensor_output=False, device=None),
        {"grid": create_grid((4, 4, 4)), "img": np.arange(8).reshape((1, 2, 2, 2)), "mode": "bilinear"},
        np.array(
            [
                [
                    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]],
                    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]],
                    [[4.0, 4.0, 5.0, 5.0], [4.0, 4.0, 5.0, 5.0], [6.0, 6.0, 7.0, 7.0], [6.0, 6.0, 7.0, 7.0]],
                    [[4.0, 4.0, 5.0, 5.0], [4.0, 4.0, 5.0, 5.0], [6.0, 6.0, 7.0, 7.0], [6.0, 6.0, 7.0, 7.0]],
                ]
            ]
        ),
    ],
]


class TestResample(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_resample(self, input_param, input_data, expected_val):
        g = Resample(**input_param)
        result = g(**input_data)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        if isinstance(result, torch.Tensor):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
