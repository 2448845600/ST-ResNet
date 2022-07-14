"""
搭建 ST-ResNet pytorch 版本
"""
import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias= True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)

        x = self.relu(x)
        x = self.conv(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False):
        super(ResUnit, self).__init__()
        self.bn_relu_conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, use_bn)
        self.bn_relu_conv2 = ConvBlock(in_channels, out_channels, kernel_size, stride, use_bn)

    def forward(self, x):
        residual = x
        output = self.bn_relu_conv1(x)
        output = self.bn_relu_conv2(output)
        output += residual
        return output


class RepeatUnits(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False, repeat_num=3):
        super(RepeatUnits, self).__init__()
        self.repeat_num = repeat_num
        res_way = [ResUnit(in_channels, out_channels, kernel_size, stride, use_bn) for _ in range(self.repeat_num)]
        self.repeat_units = nn.Sequential(*res_way)

    def forward(self, x):
        return self.repeat_units(x)


class ParametricFusion(nn.Module):
    def __init__(self, c, h, w):
        super(ParametricFusion, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, c, h, w))

    def forward(self, x):
        return self.weights * x


class STResNet(nn.Module):
    """
    C: Closeness
    P: Period
    T: Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    """
    def __init__(self,
                 c_conf=(3, 2, 32, 32),
                 p_conf=(3, 2, 32, 32),
                 t_conf=(3, 2, 32, 32),
                 external_dim=8,
                 repeat_num=3):
        super(STResNet, self).__init__()
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.external_dim = external_dim
        self.repeat_num = repeat_num

        assert self.c_conf[1:] == self.p_conf[1:] == self.t_conf[1:]
        self.channel, self.height, self.width = self.c_conf[1:]

        self.c_way = self._make_one_way(self.channel * self.c_conf[0])
        self.p_way = self._make_one_way(self.channel * self.p_conf[0])
        self.t_way = self._make_one_way(self.channel * self.t_conf[0])
        self.e_way = nn.Sequential(
            nn.Linear(self.external_dim, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, self.channel * self.height * self.width, bias=True), # 对齐输出尺寸
            nn.ReLU(),
        )
        self.tanh = torch.tanh

    def _make_one_way(self, in_channels):
        return nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=64),
            RepeatUnits(in_channels=in_channels, out_channels=64, repeat_num=self.repeat_num),
            nn.ReLU(),
            conv3x3(in_channels=64, out_channels=2),
            ParametricFusion(self.channel, self.height, self.width),
        )

    def forward(self, input_c, input_p, input_t, input_e):
        c_out = self.c_way(input_c)
        p_out = self.p_way(input_p)
        t_out = self.t_way(input_t)
        e_out = self.e_way(input_e)
        out = self.tanh(c_out + p_out + t_out + e_out)
        return out


if __name__ == '__main__':
    st_resnet = STResNet()
    print(st_resnet)
