#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        else:
            m.initialize()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1    = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out
        return self.sigmoid(out)

    def initialize(self):
        weight_init(self)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class Res_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

    def initialize(self):
        weight_init(self)


class Bkbone(nn.Module):
    def __init__(self, block,  nb_filter, block_nums):
        super(Bkbone, self).__init__()
        input_channel = 3
        self.pool    = nn.MaxPool2d(2, 2)
        self.conv0_0 = self._make_layer(block, input_channel, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], block_nums[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], block_nums[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], block_nums[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], block_nums[3])

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return x0_0, x1_0, x2_0, x3_0, x4_0

    def initialize(self):
        weight_init(self)


class Decoder1(nn.Module):
    def __init__(self, block,  nb_filter, block_nums):
        super(Decoder1, self).__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0] * 2, nb_filter[1], block_nums[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1] * 2, nb_filter[2], block_nums[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2] * 2, nb_filter[3], block_nums[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2] + nb_filter[0] * 2, nb_filter[1], block_nums[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3] + nb_filter[1] * 2, nb_filter[2], block_nums[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0] * 2, nb_filter[1], block_nums[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = input[0]
        x1_0 = input[1]
        x2_0 = input[2]
        x3_0 = input[3]
        x4_0 = input[4]

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1), self.down(x0_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1), self.down(x1_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2), self.down(x0_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1), self.down(x2_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2), self.down(x1_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return x4_0, x3_1, x2_2, x1_3, x0_4

    def initialize(self):
        weight_init(self)


class Decoder2(nn.Module):
    def __init__(self, block, nb_filter, block_nums):
        super(Decoder2, self).__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4  = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_8  = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x4_0 = input[0]
        x3_1 = input[1]
        x2_2 = input[2]
        x1_3 = input[3]
        x0_4 = input[4]

        fx4_0 = self.up_16(self.conv0_4_1x1(x4_0))
        fx3_1 = self.up_8(self.conv0_3_1x1(x3_1))
        fx2_2 = self.up_4(self.conv0_2_1x1(x2_2))
        fx1_3 = self.up(self.conv0_1_1x1(x1_3))
        fx0_4 = x0_4

        Final_x0_4 = self.conv0_4_final(torch.cat([fx4_0, fx3_1, fx2_2, fx1_3, fx0_4], 1))
        return fx4_0, fx3_1, fx2_2, fx1_3, fx0_4, Final_x0_4

    def initialize(self):
        weight_init(self)


class Decoder3(nn.Module):
    def __init__(self, block, nb_filter, block_nums):
        super(Decoder3, self).__init__()
        self.conv_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])
        self.conv4_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv3_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv2_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input1, input2):
        out0 = torch.cat([input1[0], input2[0]], dim=1)
        out1 = torch.cat([input1[1], input2[1]], dim=1)
        out2 = torch.cat([input1[2], input2[2]], dim=1)
        out3 = torch.cat([input1[3], input2[3]], dim=1)
        out4 = torch.cat([input1[4], input2[4]], dim=1)

        x0 = self.conv4_1x1(out0)
        x1 = self.conv3_1x1(out1)
        x2 = self.conv2_1x1(out2)
        x3 = self.conv1_1x1(out3)
        x4 = self.conv1_1x1(out4)

        Final_x = self.conv_final(torch.cat([x0, x1, x2, x3, x4], 1))
        return Final_x

    def initialize(self):
        weight_init(self)


class iSmall(nn.Module):
    def __init__(self, cfg, block,  nb_filter, block_nums):
        super(iSmall, self).__init__()
        self.cfg = cfg
        self.bbkbone = Bkbone(block,  nb_filter, block_nums)
        self.dbkbone = Bkbone(block, nb_filter, block_nums)
        self.decoderb1 = Decoder1(block,  nb_filter, block_nums)
        self.decoderd1 = Decoder1(block,  nb_filter, block_nums)
        self.decoderb2 = Decoder2(block,  nb_filter, block_nums)
        self.decoderd2 = Decoder2(block,  nb_filter, block_nums)
        self.decoder3 = Decoder3(block, nb_filter, block_nums)

        self.linearb = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.lineard = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(48, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, kernel_size=3, padding=1))
        self.initialize()

    def forward(self, x, shape=None):
        xb0_0, xb1_0, xb2_0, xb3_0, xb4_0 = self.bbkbone(x)
        xd0_0, xd1_0, xd2_0, xd3_0, xd4_0 = self.dbkbone(x)
        xb4_0, xb3_1, xb2_2, xb1_3, xb0_4 = self.decoderb1([xb0_0, xb1_0, xb2_0, xb3_0, xb4_0])
        xd4_0, xd3_1, xd2_2, xd1_3, xd0_4 = self.decoderd1([xd0_0, xd1_0, xd2_0, xd3_0, xd4_0])
        fxb4_0, fxb3_1, fxb2_2, fxb1_3, fxb0_4, Final_xb0_4 = self.decoderb2([xb4_0, xb3_1, xb2_2, xb1_3, xb0_4])
        fxd4_0, fxd3_1, fxd2_2, fxd1_3, fxd0_4, Final_xd0_4 = self.decoderd2([xd4_0, xd3_1, xd2_2, xd1_3, xd0_4])
        out0_4 = self.decoder3([fxb4_0, fxb3_1, fxb2_2, fxb1_3, fxb0_4], [fxd4_0, fxd3_1, fxd2_2, fxd1_3, fxd0_4])
        out0_4 = torch.cat([Final_xb0_4, Final_xd0_4, out0_4], dim=1)

        if shape is None:
            shape = x.size()[2:]
        out4 = F.interpolate(self.linear(out0_4), size=shape, mode='bilinear')
        outb4 = F.interpolate(self.linearb(Final_xb0_4), size=shape, mode='bilinear')
        outd4 = F.interpolate(self.lineard(Final_xd0_4), size=shape, mode='bilinear')
        return outb4, outd4, out4

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
