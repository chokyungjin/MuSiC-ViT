import re
import torch.nn as nn

from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.densenet import DenseNet

from ._base import EncoderMixin


class TransitionWithSkip(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        for module in self.module:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip


class DenseNetEncoder(DenseNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("DenseNet encoders do not support dilated mode "
                         "due to pooling operation for downsampling!")

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.features.conv0, self.features.norm0, self.features.relu0),
            nn.Sequential(self.features.pool0, self.features.denseblock1,
                          TransitionWithSkip(self.features.transition1)),
            nn.Sequential(self.features.denseblock2, TransitionWithSkip(self.features.transition2)),
            nn.Sequential(self.features.denseblock3, TransitionWithSkip(self.features.transition3)),
            nn.Sequential(self.features.denseblock4, self.features.norm5)
        ]

    def forward(self, x):

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if isinstance(x, (list, tuple)):
                x, skip = x
                features.append(skip)
            else:
                features.append(x)

        return features

    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")

        super().load_state_dict(state_dict)


densenet_encoders = {
    "densenet121": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet121"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 1024),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
        },
    },
    "densenet169": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet169"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1280, 1664),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 32, 32),
        },
    },
    "densenet201": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet201"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1792, 1920),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 48, 32),
        },
    },
    "densenet161": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet161"],
        "params": {
            "out_channels": (3, 96, 384, 768, 2112, 2208),
            "num_init_features": 96,
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),
        },
    },
}
