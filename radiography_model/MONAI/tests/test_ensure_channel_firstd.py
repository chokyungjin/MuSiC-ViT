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

import nibabel as nib
import numpy as np
from parameterized import parameterized
from PIL import Image

from monai.transforms import EnsureChannelFirstd, LoadImaged

TEST_CASE_1 = [{"keys": "img"}, ["test_image.nii.gz"], None]

TEST_CASE_2 = [{"keys": "img"}, ["test_image.nii.gz"], -1]

TEST_CASE_3 = [
    {"keys": "img"},
    ["test_image.nii.gz", "test_image2.nii.gz", "test_image3.nii.gz"],
    None,
]


class TestEnsureChannelFirstd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_load_nifti(self, input_param, filenames, original_channel_dim):
        if original_channel_dim is None:
            test_image = np.random.rand(128, 128, 128)
        elif original_channel_dim == -1:
            test_image = np.random.rand(128, 128, 128, 1)

        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                filenames[i] = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), filenames[i])
            result = LoadImaged(**input_param)({"img": filenames})
            result = EnsureChannelFirstd(**input_param)(result)
            self.assertEqual(result["img"].shape[0], len(filenames))

    def test_load_png(self):
        spatial_size = (256, 256, 3)
        test_image = np.random.randint(0, 256, size=spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            Image.fromarray(test_image.astype("uint8")).save(filename)
            result = LoadImaged(keys="img")({"img": filename})
            result = EnsureChannelFirstd(keys="img")(result)
            self.assertEqual(result["img"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
