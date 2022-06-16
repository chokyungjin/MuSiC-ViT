import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.networks.blocks.warp import Warp

LOW_POWER_TEST_CASES = [
    [
        {"mode": 0, "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.zeros(1, 2, 2, 2)},
        torch.arange(4).reshape((1, 1, 2, 2)),
    ],
    [
        {"mode": 1, "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 2, 2, 2)},
        torch.tensor([[[[3, 0], [0, 0]]]]),
    ],
]

HIGH_POWER_TEST_CASES = [
    [
        {"mode": 2, "padding_mode": "border"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]]),
    ],
    [
        {"mode": 3, "padding_mode": "reflection"},
        {"image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[[[7, 6], [5, 4]], [[3, 2], [1, 0]]]]]),
    ],
]

TEST_CASES = LOW_POWER_TEST_CASES
# if USE_COMPILED:
#     TEST_CASES += HIGH_POWER_TEST_CASES


class TestWarp(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_resample(self, input_param, input_data, expected_val):
        warp_layer = Warp(**input_param)
        result = warp_layer(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_ill_shape(self):
        warp_layer = Warp()
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 1, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
