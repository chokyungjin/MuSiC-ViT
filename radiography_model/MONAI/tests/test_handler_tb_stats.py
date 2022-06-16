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

from ignite.engine import Engine, Events
from torch.utils.tensorboard import SummaryWriter

from monai.handlers import TensorBoardStatsHandler


class TestHandlerTBStats(unittest.TestCase):
    def test_metrics_print(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # set up engine
            def _train_func(engine, batch):
                return batch + 1.0

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1

            # set up testing handler
            stats_handler = TensorBoardStatsHandler(log_dir=tempdir)
            stats_handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            stats_handler.close()
            # check logging output
            self.assertTrue(len(glob.glob(tempdir)) > 0)

    def test_metrics_writer(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # set up engine
            def _train_func(engine, batch):
                return batch + 1.0

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1

            # set up testing handler
            writer = SummaryWriter(log_dir=tempdir)
            stats_handler = TensorBoardStatsHandler(
                writer, output_transform=lambda x: {"loss": x * 2.0}, global_epoch_transform=lambda x: x * 3.0
            )
            stats_handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            writer.close()
            # check logging output
            self.assertTrue(len(glob.glob(tempdir)) > 0)


if __name__ == "__main__":
    unittest.main()
