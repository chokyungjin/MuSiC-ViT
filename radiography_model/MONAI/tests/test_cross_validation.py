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
import unittest
from urllib.error import ContentTooShortError, HTTPError

from monai.apps import CrossValidation, DecathlonDataset
from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityd, ToTensord
from tests.utils import skip_if_quick


class TestCrossValidation(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), 52)
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertTrue("image_meta_dict" in dataset[0])
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 34, 49, 41))

        cvdataset = CrossValidation(
            dataset_cls=DecathlonDataset,
            nfolds=5,
            seed=12345,
            root_dir=testing_dir,
            task="Task04_Hippocampus",
            section="validation",
            transform=transform,
            download=True,
        )

        try:  # will start downloading if testing_dir doesn't have the Decathlon files
            data = cvdataset.get_dataset(folds=0)
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, RuntimeError):
                # FIXME: skip MD5 check as current downloading method may fail
                self.assertTrue(str(e).startswith("md5 check"))
            return  # skipping this test due the network connection errors

        _test_dataset(data)

        # test training data for fold 0 of 5 splits
        data = cvdataset.get_dataset(folds=[1, 2, 3, 4])
        self.assertTupleEqual(data[0]["image"].shape, (1, 35, 52, 33))
        self.assertEqual(len(data), 208)
        # test train / validation for fold 4 of 5 splits
        data = cvdataset.get_dataset(folds=[4])
        self.assertTupleEqual(data[0]["image"].shape, (1, 38, 53, 30))
        self.assertEqual(len(data), 52)
        data = cvdataset.get_dataset(folds=[0, 1, 2, 3])
        self.assertTupleEqual(data[0]["image"].shape, (1, 34, 49, 41))
        self.assertEqual(len(data), 208)


if __name__ == "__main__":
    unittest.main()
