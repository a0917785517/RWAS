import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from typing import Dict
from torchsummary import summary
from thop import profile
from torch import Tensor
from collections import OrderedDict
try:
    from .custorchvision.models.mobilenetv3 import mobilenet_v3_large
    from .custorchvision.models.mobilenetv2 import mobilenet_v2_Conv
    from .custorchvision.models.efficientnetlite import build_efficientnet_lite
    from .custorchvision.models._utils import IntermediateLayerGetter
    from .custorchvision.models.segmentation import lraspp_mobilenet_v3_large
except ImportError:
    from custorchvision.models.mobilenetv3 import mobilenet_v3_large
    from custorchvision.models.mobilenetv2 import mobilenet_v2_Conv
    from custorchvision.models.efficientnetlite import build_efficientnet_lite
    from custorchvision.models._utils import IntermediateLayerGetter
    from custorchvision.models.segmentation import lraspp_mobilenet_v3_large


class Net(nn.Module):
    def __init__(
        self,
        low_channels: int = 40,
        high_channels: int = 960,
        num_classes: int = 1,
        inter_channels: int = 128,
        backbone = None,
        segmentation_head = None,
        bool_pretrained = False
    ) -> None:
        super(Net, self).__init__()
        self.special = False # Use LRASPP
        self.segmentation_head = segmentation_head
            
        if backbone == "MobileNetV3":
            net = mobilenet_v3_large(pretrained=bool_pretrained).features
            net.classifier = nn.Sequential()  # remove last fc layer
            self.backbone = net
            self.regression_head = RGHead(960, 896, 512)
           

    def forward(self, x):
        out = self.backbone(x)
        try:
            regression_out = self.regression_head(out)
        except AttributeError: # LRASPP backbone default output was fixed format output by LRASPP pretrained model.
            regression_out = self.regression_head(out['high'])

        return regression_out
    

class LRASPPHead(nn.Module):

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int
    ) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCNHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, (1, 1), padding=0, bias=False),
            nn.Conv2d(num_classes, num_classes, 1)
        )
    def forward(self, x):
        return self.block(x)
    

class RGHead(nn.Module): # Regression Head alternative Freespace Head

    def __init__(
        self,
        first_layer: int,
        second_layer: int,
        end_layer: int,
        ) -> None:
        super().__init__()
        self.fc_output = nn.Conv2d(first_layer, second_layer, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.fc_output2 = nn.Conv2d(second_layer, end_layer, kernel_size=1, bias=False)
    def forward(self, input) -> Tensor:
        out2 = F.adaptive_avg_pool2d(input, (1,1))
        out2 = self.fc_output(out2)
        out2 = self.relu(out2)
        out2 = self.fc_output2(out2)

        return out2

def test():
    net = Net(
        backbone="MobileNetV3",
        bool_pretrained = True
        ).to(torch.device("cuda"))
    summary(net, (3, 288, 512))
    
def FlopsAndParams():
    net = Net(
        backbone="MobileNetV3",
        bool_pretrained = True
        )
    input = torch.randn(1, 3, 288, 512)
    flops, params = profile(net, inputs=(input, ))
    print('flops :%.3f'%(flops/1024**3), 'G')
    print('params :%.3f'%(params/1024**2), 'MB')

if __name__ == "__main__":
    test()
    FlopsAndParams()
