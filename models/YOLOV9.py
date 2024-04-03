import torch
import torch.nn as nn

from models.common import Conv, RepNCSPELAN4, ADown, SPPELAN, DDetect


class YOLOV9(nn.Module):
    def __init__(self, C=80, deploy=True):
        super(YOLOV9, self).__init__()
        in_channels = 3

        ############ backbone ############
        # conv down
        self.conv1 = Conv(in_channels, 64, 3, 2)

        # conv down
        self.conv2 = Conv(64, 128, 3, 2)

        # elan-1 block
        self.elan1 = RepNCSPELAN4(128, 256, 128, 64, 1)

        # avg-conv down
        self.ad1 = ADown(256, 256)

        # elan-2 block
        self.elan2 = RepNCSPELAN4(256, 512, 256, 128, 1)    # 4

        # avg-conv down
        self.ad2 = ADown(512, 512)

        # elan-2 block
        self.elan3 = RepNCSPELAN4(512, 512, 512, 256, 1)    # 6

        # avg-conv down
        self.ad3 = ADown(512, 512)

        # elan-2 block
        self.elan4 = RepNCSPELAN4(512, 512, 512, 256, 1)

        ############ head ############
        # elan-spp block
        self.sppelan = SPPELAN(512, 512, 256)               # 9

        # up-concat merge
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        # elan-2 block
        self.elan5 = RepNCSPELAN4(1024, 512, 512, 256, 1)   # 12

        # up-concat merge
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        # elan-2 block
        self.elan6 = RepNCSPELAN4(1024, 256, 256, 128, 1)    # 15

        # avg-conv-down merge
        self.ad4 = ADown(256, 256)

        # elan-2 block
        self.elan7 = RepNCSPELAN4(768, 512, 512, 256, 1)    # 18

        # avg-conv-down merge
        self.ad5 = ADown(512, 512)

        # elan-2 block
        self.elan8 = RepNCSPELAN4(1024, 512, 512, 256, 1)    # 21

        # detect
        self.ddetect = DDetect(nc=C, ch=(256, 512, 512), deploy=deploy)

    def forward(self, x):
        ############ backbone ############
        # conv down
        x = self.conv2(self.conv1(x))

        # elan-1 block
        x = self.elan1(x)

        # avg-conv down
        x = self.ad1(x)

        # elan-2 block
        x_4 = self.elan2(x)

        # avg-conv down
        x = self.ad2(x_4)

        # elan-2 block
        x_6 = self.elan3(x)

        # avg-conv down
        x = self.ad3(x_6)

        # elan-2 block
        x = self.elan4(x)

        ############ head ############
        # elan-spp block
        x_9 = self.sppelan(x)

        # up-concat merge
        x = self.up1(x_9)
        x = torch.cat([x, x_6], dim=1)

        # elan-2 block
        x_12 = self.elan5(x)

        # up-concat merge
        x = self.up2(x_12)
        x = torch.cat([x, x_4], dim=1)

        # elan-2 block
        x_15 = self.elan6(x)

        # avg-conv-down merge
        x = self.ad4(x_15)
        x = torch.cat([x, x_12], dim=1)

        # elan-2 block
        x_18 = self.elan7(x)

        # avg-conv-down merge
        x = self.ad5(x_18)
        x = torch.cat([x, x_9], dim=1)

        # elan-2 block
        x_21 = self.elan8(x)

        # detect
        y = self.ddetect([x_15, x_18, x_21])
        return y