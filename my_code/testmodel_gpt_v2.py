import torch
import torch.nn as nn

# 3D版的SE注意力模块
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        reduced_channels = max(channel // reduction, 4)  # 确保通道数不小于4
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y  # 通道加权

# 使用深度可分离卷积和1x1卷积的残差块
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        super(ResidualBlock3D, self).__init__()

        # 深度可分离卷积
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, 
                                        groups=in_channels, bias=False)  # 深度卷积
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)  # 逐点卷积

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1 卷积

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.use_se = use_se

        if self.use_se:
            self.se = SELayer3D(out_channels)

    def forward(self, x):
        identity = x

        out = self.depthwise_conv(x)  # 深度卷积
        out = self.pointwise_conv(out)  # 逐点卷积
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1x1(out)  # 1x1 卷积
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)  # SE模块

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

# 主模型，结合3D CNN、残差块、批量归一化和注意力机制
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.in_channels = 60  # 输入通道数为60（梯度方向）

        channels = [32, 64, 128, 256]

        # 初始卷积层
        self.conv1 = nn.Conv3d(60, channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # 残差块层，使用深度可分离卷积和1x1卷积
        self.layer1 = self._make_layer(channels[0], channels[0], blocks=10, stride=1, use_se=True)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks=10, stride=1, use_se=True)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks=10, stride=1, use_se=True)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks=10, stride=1, use_se=True)

        # 使用全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels[3], 3)  # 输出3个NODDI指标

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
        # 输入形状：[batch_size, 60, 3, 3, 3]
        x = x.view(-1, 60, 3, 3, 3)  # 确保输入形状正确
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 输出通道：32
        x = self.layer2(x)  # 输出通道：64
        x = self.layer3(x)  # 输出通道：128
        x = self.layer4(x)  # 输出通道：256

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平为 [batch_size, channels[3]]

        x = self.fc(x)

        return x.view(x.size(0), 3, 1)
model = Mymodel()
input_tensor = torch.randn(16, 60, 27)  # 示例输入
output = model(input_tensor)
print(output.shape)  # 应输出 [16, 3, 1]