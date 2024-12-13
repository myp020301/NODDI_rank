import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D版的SE注意力模块
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)  # 通道加权

# 残差块，包含批量归一化和可选的SE模块
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer3D(out_channels)

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)  # 应用SE注意力机制

        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样以匹配尺寸

        out += identity  # 残差连接
        out = self.relu(out)

        return out

# 主模型，结合3D CNN、残差块、批量归一化和注意力机制
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.in_channels = 60  # 输入通道数为60（梯度方向）

        # 初始卷积层
        self.conv1 = nn.Conv3d(60, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # 残差块层
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, use_se=True)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=1, use_se=True)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=1, use_se=True)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=1, use_se=True)

        # 全连接层，用于回归输出
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3)  # 输出3个NODDI指标
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_se=False):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # 如果尺寸或通道数不匹配，进行下采样
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride, downsample, use_se))
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状：[batch_size, 60, 27]
        x = x.view(-1, 60, 3, 3, 3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 输出通道：64
        x = self.layer2(x)  # 输出通道：128
        x = self.layer3(x)  # 输出通道：256
        x = self.layer4(x)  # 输出通道：512

        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        
        return x.view(x.size(0), 3, 1)  # 输出形状：[batch_size, 3, 1]

# 示例
# model = NODDINet()
# print(model)
