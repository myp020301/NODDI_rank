import torch
import torch.nn as nn

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)

        # 如果输入输出维度不同，需要调整shortcut
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        identity = self.shortcut(identity)
        out += identity  # 残差连接
        out = self.relu(out)
        return out

# 定义模型
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # 输入层：从 [batchsize, 60, 27] 转换为 [batchsize, 300, 300]
        self.fc_input_1 = nn.Linear(27, 300)
        self.fc_input_2 = nn.Linear(60, 300)
        self.relu = nn.ReLU(inplace=True)

        # 定义模型的层次结构
        self.layers = nn.Sequential(
            # 重复多个残差块，参数量达到您提供的模型水平
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            ResidualBlock(300, 300),
            # 可以根据需要继续添加更多的残差块
        )

        # 减少特征维度
        self.reduce_layers_left = nn.Sequential(
            ResidualBlock(300, 150),
            ResidualBlock(150, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 3)
        )
        
        self.reduce_layers_right = nn.Sequential(
            ResidualBlock(300, 150),
            ResidualBlock(150, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 75),
            ResidualBlock(75, 1)
        )

    def forward(self, x):
        # x: [batchsize, 60, 27]
        x = self.fc_input_1(x)  # [batchsize, 60, 300]
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 调整维度到 [batchsize, 300, 60]
        x = self.fc_input_2(x)  # [batchsize, 300, 300]
        x = self.relu(x)

        x = self.layers(x)  # 通过残差块序列

        x = self.reduce_layers_left(x)  # 减少特征维度
        x =  x.permute(0, 2, 1)
        x = self.reduce_layers_right(x)  
        # 输出形状：[batchsize, 3, 1]
        return x

# model = Mymodel()
# input_tensor = torch.randn(16, 60, 27)  # 示例输入
# output = model(input_tensor)
# print(output.shape)  # 应输出 [16, 3, 1]