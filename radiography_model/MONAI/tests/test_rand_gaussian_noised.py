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

from monai.transforms import RandGaussianNoised
from tests.utils import NumpyImageTestCase2D, TorchImageTestCase2D

TEST_CASE_0 = ["test_zero_mean", ["img1", "img2"], 0, 0.1]
TEST_CASE_1 = ["test_non_zero_mean", ["img1", "img2"], 1, 0.5]
TEST_CASES = [TEST_CASE_0, TEST_CASE_1]

seed = 0


def test_numpy_or_torch(keys, mean, std, imt):
    gaussian_fn = RandGaussianNoised(keys=keys, prob=1.0, mean=mean, std=std)
    gaussian_fn.set_random_state(seed)
    noised = gaussian_fn({k: imt for k in keys})
    np.random.seed(seed)
    np.random.random()
    for k in keys:
        expected = imt + np.random.normal(mean, np.random.uniform(0, std), size=imt.shape)
        np.testing.assert_allclose(expected, noised[k], atol=1e-5, rtol=1e-5)


# Test with numpy
class TestRandGaussianNoisedNumpy(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES)
    def test_correct_results(self, _, keys, mean, std):
        test_numpy_or_torch(keys, mean, std, self.imt)


# Test with torch
class TestRandGaussianNoisedTorch(TorchImageTestCase2D):
    @parameterized.expand(TEST_CASES)
    def test_correct_results(self, _, keys, mean, std):
        test_numpy_or_torch(keys, mean, std, self.imt)


if __name__ == "__main__":
    unittest.main()
