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

import warnings
from typing import Union

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction


class DiceMetric:
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input `y_pred` (BNHW[D] where N is number of classes) is compared with ground truth `y` (BNHW[D]).
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` for an instance of DiceLoss to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background so excluding it in such cases helps convergence.

    Args:
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred is not a binarized tensor here!")
        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute dice (BxC) for each channel for each batch
        f = compute_meandice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
        )

        # do metric reduction
        f, not_nans = do_metric_reduction(f, self.reduction)
        return f, not_nans


def compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """

    if not include_background:
        y_pred, y = ignore_background(
            y_pred=y_pred,
            y=y,
        )

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    f = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_o.device))
    return f  # returns array of Dice with shape: [batch, n_classes]
