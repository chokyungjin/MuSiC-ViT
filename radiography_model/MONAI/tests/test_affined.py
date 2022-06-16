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

from monai.transforms import Affined

TEST_CASES = [
    [
        dict(keys="img", padding_mode="zeros", as_tensor_output=False, spatial_size=(-1, 0), device=None),
        {"img": np.arange(9).reshape((1, 3, 3))},
        np.arange(9).reshape(1, 3, 3),
    ],
    [
        dict(keys="img", padding_mode="zeros", as_tensor_output=False, device=None),
        {"img": np.arange(4).reshape((1, 2, 2))},
        np.arange(4).reshape(1, 2, 2),
    ],
    [
        dict(keys="img", padding_mode="zeros", spatial_size=(4, 4), as_tensor_output=False, device=None),
        {"img": np.arange(4).reshape((1, 2, 2))},
        np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
    ],
    [
        dict(
            keys="img",
            rotate_params=[np.pi / 2],
            padding_mode="zeros",
            spatial_size=(4, 4),
            as_tensor_output=False,
            device=None,
        ),
        {"img": np.arange(4).reshape((1, 2, 2))},
        np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
    ],
    [
        dict(keys="img", padding_mode="zeros", spatial_size=(-1, 0, 0), as_tensor_output=False, device=None),
        {"img": np.arange(27).reshape((1, 3, 3, 3))},
        np.arange(27).reshape(1, 3, 3, 3),
    ],
    [
        dict(keys="img", padding_mode="zeros", spatial_size=(4, 4, 4), as_tensor_output=False, device=None),
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
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
        dict(
            keys="img",
            rotate_params=[np.pi / 2],
            padding_mode="zeros",
            spatial_size=(4, 4, 4),
            as_tensor_output=False,
            device=None,
        ),
        {"img": np.arange(8).reshape((1, 2, 2, 2))},
        np.array(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 6.0, 4.0, 0.0], [0.0, 7.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        ),
    ],
]


class TestAffined(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_affine(self, input_param, input_data, expected_val):
        g = Affined(**input_param)
        result = g(input_data)["img"]
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
