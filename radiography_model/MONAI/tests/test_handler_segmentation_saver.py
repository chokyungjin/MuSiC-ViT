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

import os
import tempfile
import unittest

import numpy as np
import torch
from ignite.engine import Engine
from parameterized import parameterized

from monai.handlers import SegmentationSaver

TEST_CASE_0 = [".nii.gz"]

TEST_CASE_1 = [".png"]


class TestHandlerSegmentationSaver(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    def test_saved_content(self, output_ext):
        with tempfile.TemporaryDirectory() as tempdir:

            # set up engine
            def _train_func(engine, batch):
                return torch.randint(0, 255, (8, 1, 2, 2)).float()

            engine = Engine(_train_func)

            # set up testing handler
            saver = SegmentationSaver(output_dir=tempdir, output_postfix="seg", output_ext=output_ext, scale=255)
            saver.attach(engine)

            data = [{"filename_or_obj": ["testfile" + str(i) + ".nii.gz" for i in range(8)]}]
            engine.run(data, max_epochs=1)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg" + output_ext)
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    def test_save_resized_content(self, output_ext):
        with tempfile.TemporaryDirectory() as tempdir:

            # set up engine
            def _train_func(engine, batch):
                return torch.randint(0, 255, (8, 1, 2, 2)).float()

            engine = Engine(_train_func)

            # set up testing handler
            saver = SegmentationSaver(output_dir=tempdir, output_postfix="seg", output_ext=output_ext, scale=255)
            saver.attach(engine)

            data = [
                {
                    "filename_or_obj": ["testfile" + str(i) + ".nii.gz" for i in range(8)],
                    "spatial_shape": [(28, 28)] * 8,
                    "affine": [np.diag(np.ones(4)) * 5] * 8,
                    "original_affine": [np.diag(np.ones(4)) * 1.0] * 8,
                }
            ]
            engine.run(data, max_epochs=1)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg" + output_ext)
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))


if __name__ == "__main__":
    unittest.main()
