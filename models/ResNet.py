import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    针对 Res18 和 Res34 的残差结构
    """
    expansion = 1 # 主分支中卷积核个数是否变化，18和34中是一摸一样的

    # downsample是shortcut是否需要 1x1 卷积操作，每个ResBlock的第一个需要
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 每个ResBlock的第一个需要将 stride 设置为2（传入参数），这样才能下采样（虚线连接）
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 在卷积之后激活之前使用BatchNorm，参数是输入的深度
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 第二个卷积层的 stride 都是1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 针对 shortcut 使用的，不是第一个的话都是None
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        # 主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 加上shortcut，再激活relu
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    针对 Res50 和 Res101 和 Res152 的残差结构
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4 # 主分支卷积核通道的变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        # 例子：in是256，out是64
        super(Bottleneck, self).__init__()
        # width理解是out_channel
        width = int(out_channel * (width_per_group / 64.)) * groups

        # 256 -> 64
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 下面stride使用的是传入的参数
        # 64 -> 64
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 64 -> 256（乘了expansion）
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # 先加上再激活
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block, # 对应的残差结构 BasicBlock BottleNeck
                 blocks_num, # 残差结构的数目，是一个列表，如 34 -> [3, 4, 6, 3]
                 num_classes=1000, # 分类数目
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        # 已经通过conv1和maxpool
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 第一个卷积 7x7
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 对应 conv2_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # 对应 conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # 对应 conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # 对应 conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 自适应平均池化，无论输入高宽，结果是1 1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 宽高都是1，只有通道
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 构建Res2，Res3...
    def _make_layer(self, block, channel, block_num, stride=1):
        # block是基本模块，channel是卷积层卷积核个数（第一个），block_num是包含多少个残差结构
        downsample = None
        # 对应18和34来说，后面的判断不满足
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 1x1卷积
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        # 第一个是需要判断的
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        # 从第二层开始，都是实线的残差连接
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)