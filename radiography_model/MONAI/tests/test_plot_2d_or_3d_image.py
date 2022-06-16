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

import glob
import tempfile
import unittest

import torch
from parameterized import parameterized
from torch.utils.tensorboard import SummaryWriter

from monai.visualize import plot_2d_or_3d_image

TEST_CASE_1 = [(1, 1, 10, 10)]

TEST_CASE_2 = [(1, 3, 10, 10)]

TEST_CASE_3 = [(1, 4, 10, 10)]

TEST_CASE_4 = [(1, 1, 10, 10, 10)]

TEST_CASE_5 = [(1, 3, 10, 10, 10)]


class TestPlot2dOr3dImage(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_tb_image_shape(self, shape):
        with tempfile.TemporaryDirectory() as tempdir:
            writer = SummaryWriter(log_dir=tempdir)
            plot_2d_or_3d_image(torch.zeros(shape), 0, writer)
            writer.flush()
            writer.close()
            self.assertTrue(len(glob.glob(tempdir)) > 0)


if __name__ == "__main__":
    unittest.main()
