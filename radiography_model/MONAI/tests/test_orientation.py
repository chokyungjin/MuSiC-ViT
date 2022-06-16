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

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.transforms import Orientation, create_rotate, create_translate

TEST_CASES = [
    [
        {"axcodes": "RAS"},
        np.arange(12).reshape((2, 1, 2, 3)),
        {"affine": np.eye(4)},
        np.arange(12).reshape((2, 1, 2, 3)),
        "RAS",
    ],
    [
        {"axcodes": "ALS"},
        np.arange(12).reshape((2, 1, 2, 3)),
        {"affine": np.diag([-1, -1, 1, 1])},
        np.array([[[[3, 4, 5]], [[0, 1, 2]]], [[[9, 10, 11]], [[6, 7, 8]]]]),
        "ALS",
    ],
    [
        {"axcodes": "RAS"},
        np.arange(12).reshape((2, 1, 2, 3)),
        {"affine": np.diag([-1, -1, 1, 1])},
        np.array([[[[3, 4, 5], [0, 1, 2]]], [[[9, 10, 11], [6, 7, 8]]]]),
        "RAS",
    ],
    [
        {"axcodes": "AL"},
        np.arange(6).reshape((2, 1, 3)),
        {"affine": np.eye(3)},
        np.array([[[0], [1], [2]], [[3], [4], [5]]]),
        "AL",
    ],
    [{"axcodes": "L"}, np.arange(6).reshape((2, 3)), {"affine": np.eye(2)}, np.array([[2, 1, 0], [5, 4, 3]]), "L"],
    [{"axcodes": "L"}, np.arange(6).reshape((2, 3)), {"affine": np.eye(2)}, np.array([[2, 1, 0], [5, 4, 3]]), "L"],
    [{"axcodes": "L"}, np.arange(6).reshape((2, 3)), {"affine": np.diag([-1, 1])}, np.arange(6).reshape((2, 3)), "L"],
    [
        {"axcodes": "LPS"},
        np.arange(12).reshape((2, 1, 2, 3)),
        {
            "affine": create_translate(3, (10, 20, 30))
            @ create_rotate(3, (np.pi / 2, np.pi / 2, np.pi / 4))
            @ np.diag([-1, 1, 1, 1])
        },
        np.array([[[[2, 5]], [[1, 4]], [[0, 3]]], [[[8, 11]], [[7, 10]], [[6, 9]]]]),
        "LPS",
    ],
    [
        {"as_closest_canonical": True},
        np.arange(12).reshape((2, 1, 2, 3)),
        {
            "affine": create_translate(3, (10, 20, 30))
            @ create_rotate(3, (np.pi / 2, np.pi / 2, np.pi / 4))
            @ np.diag([-1, 1, 1, 1])
        },
        np.array([[[[0, 3]], [[1, 4]], [[2, 5]]], [[[6, 9]], [[7, 10]], [[8, 11]]]]),
        "RAS",
    ],
    [
        {"as_closest_canonical": True},
        np.arange(6).reshape((1, 2, 3)),
        {"affine": create_translate(2, (10, 20)) @ create_rotate(2, (np.pi / 3)) @ np.diag([-1, -0.2, 1])},
        np.array([[[3, 0], [4, 1], [5, 2]]]),
        "RA",
    ],
    [
        {"axcodes": "LP"},
        np.arange(6).reshape((1, 2, 3)),
        {"affine": create_translate(2, (10, 20)) @ create_rotate(2, (np.pi / 3)) @ np.diag([-1, -0.2, 1])},
        np.array([[[2, 5], [1, 4], [0, 3]]]),
        "LP",
    ],
    [
        {"axcodes": "LPID", "labels": tuple(zip("LPIC", "RASD"))},
        np.zeros((1, 2, 3, 4, 5)),
        {"affine": np.diag([-1, -0.2, -1, 1, 1])},
        np.zeros((1, 2, 3, 4, 5)),
        "LPID",
    ],
    [
        {"as_closest_canonical": True, "labels": tuple(zip("LPIC", "RASD"))},
        np.zeros((1, 2, 3, 4, 5)),
        {"affine": np.diag([-1, -0.2, -1, 1, 1])},
        np.zeros((1, 2, 3, 4, 5)),
        "RASD",
    ],
]

ILL_CASES = [
    # no axcodes or as_cloest_canonical
    [{}, np.arange(6).reshape((2, 3)), "L"],
    # too short axcodes
    [{"axcodes": "RA"}, np.arange(12).reshape((2, 1, 2, 3)), {"affine": np.eye(4)}],
]


class TestOrientationCase(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_ornt(self, init_param, img, data_param, expected_data, expected_code):
        ornt = Orientation(**init_param)
        res = ornt(img, **data_param)
        np.testing.assert_allclose(res[0], expected_data)
        original_affine = data_param["affine"]
        np.testing.assert_allclose(original_affine, res[1])
        new_code = nib.orientations.aff2axcodes(res[2], labels=ornt.labels)
        self.assertEqual("".join(new_code), expected_code)

    @parameterized.expand(ILL_CASES)
    def test_bad_params(self, init_param, img, data_param):
        with self.assertRaises(ValueError):
            Orientation(**init_param)(img, **data_param)


if __name__ == "__main__":
    unittest.main()
