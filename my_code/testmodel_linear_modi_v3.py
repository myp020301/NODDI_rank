import torch
import torch.nn as nn

# 自定义 Lambda 层，用于在模型中插入任意操作
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# 定义残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)

        # 如果输入输出维度不同，需要调整 shortcut
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

# 定义主模型
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # 扩展阶段：将 [batchsize, 60, 27] 转换为 [batchsize, 64, 64]
        self.expand = nn.Sequential(
            nn.Linear(27, 64),  # 将最后一个维度从 27 转换为 64
            nn.ReLU(),
            LambdaLayer(lambda x: x.permute(0, 2, 1)),  # 调整维度到 [batchsize, 64, 60]
            nn.Linear(60, 64),  # 将第二个维度从 60 转换为 64
            nn.ReLU()
        )
        
        # 残差块序列，进一步增加模型复杂度
        self.residual_expand = nn.Sequential(*[nn.Sequential(ResidualBlock(64, 64), nn.ReLU()) for _ in range(10)])
        
        # 合并维度并减少：从 [batchsize, 64, 64] 到 [batchsize*64, 64]
        # 然后逐步减少到 [batchsize*64, 1]
        self.reduce_layers = nn.Sequential(
            *[nn.Sequential(ResidualBlock(64, 32), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(32, 32), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(32, 16), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(16, 16), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(16, 8), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(8, 8), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(8, 1), nn.ReLU()) for _ in range(1)]
        )
        
        # 恢复并进一步减少：从 [batchsize*64, 1] 恢复为 [batchsize, 64]
        # 然后逐步减少到 [batchsize, 3]
        self.final_reduce = nn.Sequential(
            *[nn.Sequential(ResidualBlock(64, 32), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(32, 32), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(32, 16), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(16, 16), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(16, 8), nn.ReLU()) for _ in range(1)],
            *[nn.Sequential(ResidualBlock(8, 8), nn.ReLU()) for _ in range(10)],
            *[nn.Sequential(ResidualBlock(8, 3), nn.ReLU()) for _ in range(1)]
        )
        
    def forward(self, x):
        # 输入 x: [batchsize, 60, 27]
        x = self.expand(x)  # 扩展阶段后 x: [batchsize, 64, 64]
        x = self.residual_expand(x)  # [batchsize, 64, 64]
        
        batchsize, dim1, dim2 = x.size()  # dim1=64, dim2=64
        x = x.view(batchsize * dim1, dim2)  # 合并维度 x: [batchsize*64, 64]
        x = self.reduce_layers(x)  # 减少维度 x: [batchsize*64, 1]
        
        x = x.view(batchsize, dim1)  # 恢复维度 x: [batchsize, 64]
        x = self.final_reduce(x)  # 进一步减少 x: [batchsize, 3]
        
        return x.view(batchsize, 3, 1)  # 最终输出 [batchsize, 3, 1]


#model = Mymodel()
#
#input_tensor = torch.randn(16, 60, 27)  # 示例输入 [batchsize=16, 60, 27]
#output = model(input_tensor)
#print(output.shape)  # 应输出 [16, 3, 1]