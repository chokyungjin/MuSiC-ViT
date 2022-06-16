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

from monai.transforms import ConcatItemsd


class TestConcatItemsd(unittest.TestCase):
    def test_tensor_values(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
        input_data = {
            "img1": torch.tensor([[0, 1], [1, 2]], device=device),
            "img2": torch.tensor([[0, 1], [1, 2]], device=device),
        }
        result = ConcatItemsd(keys=["img1", "img2"], name="cat_img")(input_data)
        self.assertTrue("cat_img" in result)
        result["cat_img"] += 1
        torch.testing.assert_allclose(result["img1"], torch.tensor([[0, 1], [1, 2]], device=device))
        torch.testing.assert_allclose(result["cat_img"], torch.tensor([[1, 2], [2, 3], [1, 2], [2, 3]], device=device))

    def test_numpy_values(self):
        input_data = {"img1": np.array([[0, 1], [1, 2]]), "img2": np.array([[0, 1], [1, 2]])}
        result = ConcatItemsd(keys=["img1", "img2"], name="cat_img")(input_data)
        self.assertTrue("cat_img" in result)
        result["cat_img"] += 1
        np.testing.assert_allclose(result["img1"], np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(result["cat_img"], np.array([[1, 2], [2, 3], [1, 2], [2, 3]]))


if __name__ == "__main__":
    unittest.main()
