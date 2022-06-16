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

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

from monai.handlers.utils import string_list_all_gather, write_metrics_reports
from monai.utils import ImageMetaKey as Key
from monai.utils import ensure_tuple, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
idist, _ = optional_import("ignite", "0.4.4", exact_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class MetricsSaver:
    """
    ignite handler to save metrics values and details into expected files.

    Args:
        save_dir: directory to save the metrics and metric details.
        metrics: expected final metrics to save into files, can be: None, "*" or list of strings.
            None - don't save any metrics into files.
            "*" - save all the existing metrics in `engine.state.metrics` dict into separate files.
            list of strings - specify the expected metrics to save.
            default to "*" to save all the metrics into `metrics.csv`.
        metric_details: expected metric details to save into files, the data comes from
            `engine.state.metric_details`, which should be provided by different `Metrics`,
            typically, it's some intermediate values in metric computation.
            for example: mean dice of every channel of every image in the validation dataset.
            it must contain at least 2 dims: (batch, classes, ...),
            if not, will unsequeeze to 2 dims.
            this arg can be: None, "*" or list of strings.
            None - don't save any metric_details into files.
            "*" - save all the existing metric_details in `engine.state.metric_details` dict into separate files.
            list of strings - specify the metric_details of expected metrics to save.
            if not None, every metric_details array will save a separate `{metric name}_raw.csv` file.
        batch_transform: callable function to extract the meta_dict from input batch data if saving metric details.
            used to extract filenames from input dict data.
        summary_ops: expected computation operations to generate the summary report based on specified metric_details.
            it can be: None, "*" or list of strings.
            None - don't generate summary report for every specified metric_details
            "*" - generate summary report for every metric_details with all the supported operations.
            list of strings - generate summary report for every metric_details with specified operations, they
            should be within this list: [`mean`, `median`, `max`, `min`, `90percent`, `std`].
            default to None.
        save_rank: only the handler on specified rank will save to files in multi-gpus validation, default to 0.
        delimiter: the delimiter character in CSV file, default to "\t".
        output_type: expected output file type, supported types: ["csv"], default to "csv".

    """

    def __init__(
        self,
        save_dir: str,
        metrics: Optional[Union[str, Sequence[str]]] = "*",
        metric_details: Optional[Union[str, Sequence[str]]] = None,
        batch_transform: Callable = lambda x: x,
        summary_ops: Optional[Union[str, Sequence[str]]] = None,
        save_rank: int = 0,
        delimiter: str = "\t",
        output_type: str = "csv",
    ) -> None:
        self.save_dir = save_dir
        self.metrics = ensure_tuple(metrics) if metrics is not None else None
        self.metric_details = ensure_tuple(metric_details) if metric_details is not None else None
        self.batch_transform = batch_transform
        self.summary_ops = ensure_tuple(summary_ops) if summary_ops is not None else None
        self.save_rank = save_rank
        self.deli = delimiter
        self.output_type = output_type
        self._filenames: List[str] = []

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self._started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._get_filenames)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def _started(self, engine: Engine) -> None:
        self._filenames = []

    def _get_filenames(self, engine: Engine) -> None:
        if self.metric_details is not None:
            _filenames = list(ensure_tuple(self.batch_transform(engine.state.batch)[Key.FILENAME_OR_OBJ]))
            self._filenames += _filenames

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        ws = idist.get_world_size()
        if self.save_rank >= ws:
            raise ValueError("target rank is greater than the distributed group size.")

        # all gather file names across ranks
        _images = string_list_all_gather(strings=self._filenames) if ws > 1 else self._filenames

        # only save metrics to file in specified rank
        if idist.get_rank() == self.save_rank:
            _metrics = {}
            if self.metrics is not None and len(engine.state.metrics) > 0:
                _metrics = {k: v for k, v in engine.state.metrics.items() if k in self.metrics or "*" in self.metrics}
            _metric_details = {}
            if self.metric_details is not None and len(engine.state.metric_details) > 0:
                for k, v in engine.state.metric_details.items():
                    if k in self.metric_details or "*" in self.metric_details:
                        _metric_details[k] = v

            write_metrics_reports(
                save_dir=self.save_dir,
                images=_images,
                metrics=_metrics,
                metric_details=_metric_details,
                summary_ops=self.summary_ops,
                deli=self.deli,
                output_type=self.output_type,
            )
