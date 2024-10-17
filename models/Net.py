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
        if backbone == "LRASPP":
            net = lraspp_mobilenet_v3_large(pretrained=bool_pretrained)
            net.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)
                
            self.backbone = net.backbone
            if bool_pretrained: # training will here
                self.segmentation_head = net.classifier
                self.special = True # Use LRASPP
            else: # detect will here because bool_pretrained is False
                self.segmentation_head = None
                self.special = False # Use LRASPP
            self.regression_head = RGHead(960, 896, 512)
            
        elif backbone == "MobileNetV3":
            net = mobilenet_v3_large(pretrained=bool_pretrained).features
            if segmentation_head == "LRASPP":
                stage_indices = [0] + [i for i, b in enumerate(net) if getattr(b, "_is_cn", False)] + [len(net) - 1]
                self.low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
                self.high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
                self.backbone = IntermediateLayerGetter(net, return_layers={str(self.low_pos): 'low', str(self.high_pos): 'high'})
                self.segmentation_head = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)
                self.special = True # Use LRASPP
            else:
                net.classifier = nn.Sequential()  # remove last fc layer
                self.backbone = net
            self.regression_head = RGHead(960, 896, 512)
        elif backbone == "MobileNetV2":
            net = mobilenet_v2_Conv(pretrained=bool_pretrained)
            num_ftrs = net.classifier[1].in_features # get mobilenet.classifier in_features number. int 1280
            if segmentation_head == "FCN":
                pass
            else:
                net.classifier = nn.Sequential()  # remove last fc layer
                self.backbone = net
            self.regression_head = RGHead(num_ftrs, 896, 512)
        elif backbone == "EfficientNetLite": 
            net = build_efficientnet_lite('efficientnet_lite0', 1000)
            if bool_pretrained:
                net.load_pretrain("./Pretrain_Model/EfficientNet-Lite/efficientnet_lite0.pth")
            net.avgpool = nn.Sequential()
            net.dropout = nn.Sequential()
            net.fc = nn.Sequential()
            self.backbone = net 
            if segmentation_head == "FCN":
                self.segmentation_head = FCNHead(1280, 1)
            else:
                self.segmentation_head = False
            self.regression_head = RGHead(1280, 896, 512)
        else:
            raise Exception("Please set a backbone type ! !")
           

    def forward(self, x):
        if self.segmentation_head:
            if self.special:
                out = self.backbone(x)
                segmentation_out = self.segmentation_head(out)
                segmentation_out = F.interpolate(segmentation_out, size=x.shape[-2:], mode='bilinear', align_corners=True)
                regression_out = self.regression_head(out['high'])
            else:
                out = self.backbone(x)
                segmentation_out = self.segmentation_head(out)
                segmentation_out = F.interpolate(segmentation_out, size=x.shape[-2:], mode='bilinear', align_corners=True)
                regression_out = self.regression_head(out)

            return segmentation_out, regression_out

        else:
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
        backbone="LRASPP",
        segmentation_head = "LRASPP",
        bool_pretrained = True
        ).to(torch.device("cuda"))
    summary(net, (3, 288, 512))
    
def FlopsAndParams():
    net = Net(
        backbone="LRASPP",
        segmentation_head = "LRASPP",
        bool_pretrained = True
        )
    input = torch.randn(1, 3, 288, 512)
    flops, params = profile(net, inputs=(input, ))
    print('flops :%.3f'%(flops/1024**3), 'G')
    print('params :%.3f'%(params/1024**2), 'MB')

if __name__ == "__main__":
    test()
    FlopsAndParams()
