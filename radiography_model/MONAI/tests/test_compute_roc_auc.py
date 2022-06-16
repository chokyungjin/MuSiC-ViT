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

from monai.metrics import compute_roc_auc
from monai.transforms import Activations, AsDiscrete

TEST_CASE_1 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[0], [1], [0], [1]]),
    True,
    True,
    "macro",
    0.75,
]

TEST_CASE_2 = [
    torch.tensor([[0.5], [0.5], [0.2], [8.3]]),
    torch.tensor([[0], [1], [0], [1]]),
    False,
    False,
    "macro",
    0.875,
]

TEST_CASE_3 = [
    torch.tensor([[0.5], [0.5], [0.2], [8.3]]),
    torch.tensor([0, 1, 0, 1]),
    False,
    False,
    "macro",
    0.875,
]

TEST_CASE_4 = [
    torch.tensor([0.5, 0.5, 0.2, 8.3]),
    torch.tensor([0, 1, 0, 1]),
    False,
    False,
    "macro",
    0.875,
]

TEST_CASE_5 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]]),
    torch.tensor([[0], [1], [0], [1]]),
    True,
    True,
    "none",
    [0.75, 0.75],
]

TEST_CASE_6 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]]),
    torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]]),
    True,
    False,
    "weighted",
    0.56667,
]

TEST_CASE_7 = [
    torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]]),
    torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]]),
    True,
    False,
    "micro",
    0.62,
]


class TestComputeROCAUC(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_value(self, y_pred, y, softmax, to_onehot, average, expected_value):
        y_pred = Activations(softmax=softmax)(y_pred)
        y = AsDiscrete(to_onehot=to_onehot, n_classes=2)(y)
        result = compute_roc_auc(y_pred=y_pred, y=y, average=average)
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
