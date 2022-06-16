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

import torch
from ignite.engine import Events

from monai.engines import EnsembleEvaluator


class TestEnsembleEvaluator(unittest.TestCase):
    def test_content(self):
        device = torch.device("cpu:0")

        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return {"image": torch.tensor([index]), "label": torch.zeros(1)}

        val_loader = torch.utils.data.DataLoader(TestDataset())

        class TestNet(torch.nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        net0 = TestNet(lambda x: x + 1)
        net1 = TestNet(lambda x: x + 2)
        net2 = TestNet(lambda x: x + 3)
        net3 = TestNet(lambda x: x + 4)
        net4 = TestNet(lambda x: x + 5)

        val_engine = EnsembleEvaluator(
            device=device,
            val_data_loader=val_loader,
            networks=[net0, net1, net2, net3, net4],
            pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        )

        @val_engine.on(Events.ITERATION_COMPLETED)
        def run_post_transform(engine):
            for i in range(5):
                expected_value = engine.state.iteration + i
                torch.testing.assert_allclose(engine.state.output[f"pred{i}"], torch.tensor([[expected_value]]))

        val_engine.run()


if __name__ == "__main__":
    unittest.main()
