import math

import torch
from torch import nn


def caffe_bn_momentum(moving_average_fraction, updates):
    if updates < 1:
        raise ValueError("bn update before the first batch")
    maf = moving_average_fraction
    if maf == 1.0:
        scale = float(updates)
    else:
        scale = (1.0 - maf**updates) / (1.0 - maf)
    return 1.0 / scale


class CaffeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, moving_average_fraction=0.999, eps=1e-5):
        super().__init__(
            num_features,
            eps=eps,
            momentum=1.0,
            affine=True,
            track_running_stats=True,
        )
        self.moving_average_fraction = moving_average_fraction
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self._updates = 0

    def restore(self):
        self._updates = int(self.updates.item())

    def stash(self):
        self.updates.fill_(self._updates)

    def forward(self, input):
        if self.training and self.track_running_stats:
            self._updates += 1
            self.momentum = caffe_bn_momentum(self.moving_average_fraction, self._updates)
        return super().forward(input)


def _bn(channels, moving_average_fraction):
    return CaffeBatchNorm2d(channels, moving_average_fraction=moving_average_fraction)


class BasicBlock(nn.Module):
    def __init__(self, channels, bn_maf):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn1 = _bn(channels, bn_maf)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn2 = _bn(channels, bn_maf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels, bn_maf):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=True)
        self.bn1 = _bn(channels, bn_maf)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True)
        self.bn2 = _bn(channels, bn_maf)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.pool(x)
        if shortcut.shape[-2:] != out.shape[-2:]:
            raise RuntimeError(f"shortcut {tuple(shortcut.shape)} != conv {tuple(out.shape)}")
        out = self.relu(out + shortcut)
        zeros = torch.zeros_like(out)
        return torch.cat([out, zeros], dim=1)


class ResNet56(nn.Module):
    def __init__(self, bn_maf, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.bn = _bn(16, bn_maf)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = nn.Sequential(*[BasicBlock(16, bn_maf) for _ in range(9)])
        self.down2 = DownsampleBlock(16, bn_maf)
        self.stage2 = nn.Sequential(*[BasicBlock(32, bn_maf) for _ in range(8)])
        self.down3 = DownsampleBlock(32, bn_maf)
        self.stage3 = nn.Sequential(*[BasicBlock(64, bn_maf) for _ in range(8)])
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, CaffeBatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.updates.zero_()
                module._updates = 0
            elif isinstance(module, nn.Linear):
                bound = math.sqrt(3.0 / module.weight.size(1))
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def trace(self, x):
        shapes = {}
        x = self.relu(self.bn(self.conv(x)))
        shapes["stem"] = tuple(x.shape)
        x = self.stage1(x)
        shapes["stage1"] = tuple(x.shape)
        x = self.down2(x)
        shapes["down2"] = tuple(x.shape)
        x = self.stage2(x)
        shapes["stage2"] = tuple(x.shape)
        x = self.down3(x)
        shapes["down3"] = tuple(x.shape)
        x = self.stage3(x)
        shapes["stage3"] = tuple(x.shape)
        x = self.pool(x)
        shapes["pool"] = tuple(x.shape)
        logits = self.fc(torch.flatten(x, 1))
        shapes["logits"] = tuple(logits.shape)
        return shapes

    def stash_bn(self):
        for module in self.modules():
            if isinstance(module, CaffeBatchNorm2d):
                module.stash()

    def restore_bn(self):
        for module in self.modules():
            if isinstance(module, CaffeBatchNorm2d):
                module.restore()


EXPECTED_SHAPES = {
    "stem": (1, 16, 32, 32),
    "stage1": (1, 16, 32, 32),
    "down2": (1, 32, 16, 16),
    "stage2": (1, 32, 16, 16),
    "down3": (1, 64, 8, 8),
    "stage3": (1, 64, 8, 8),
    "pool": (1, 64, 1, 1),
    "logits": (1, 10),
}
