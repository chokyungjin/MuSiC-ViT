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

from monai.data.utils import create_file_basename


class TestFilename(unittest.TestCase):
    def test_value(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output_tmp = os.path.join(tempdir, "output")
            result = create_file_basename("", "test.txt", output_tmp, "")
            expected = os.path.join(output_tmp, "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", os.path.join("foo", "test.txt"), output_tmp, "")
            expected = os.path.join(output_tmp, "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", os.path.join("foo", "test.txt"), output_tmp, "foo")
            expected = os.path.join(output_tmp, "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", os.path.join("foo", "bar", "test.txt"), output_tmp, "foo")
            expected = os.path.join(output_tmp, "bar", "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", os.path.join("foo", "bar", "test.txt"), output_tmp, "bar")
            expected = os.path.join(tempdir, "foo", "bar", "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", os.path.join("rest", "test.txt"), output_tmp, "rest")
            expected = os.path.join(tempdir, "output", "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("", "test.txt", output_tmp, "foo")
            expected = os.path.join(output_tmp, "test", "test")
            self.assertEqual(result, expected)

            result = create_file_basename("post", "test.tar.gz", output_tmp, "foo")
            expected = os.path.join(output_tmp, "test", "test_post")
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
