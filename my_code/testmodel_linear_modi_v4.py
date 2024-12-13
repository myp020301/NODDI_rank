import torch
import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = out + residual  # 残差连接
        return out
  
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # First part: Expanding 60x27 to 64x64
        self.expand = nn.Sequential(
            nn.Linear(27, 64),  # [batch, 60, 64]
            nn.ReLU(),
            nn.Linear(64, 128) ,
            nn.ReLU(),
            nn.Linear(128, 256) ,
            nn.ReLU(),
            nn.Linear(256, 512) ,
            nn.ReLU(),
            LambdaLayer(lambda x: x.permute(0, 2, 1)),  # Permute to [batch, 64, 60]
            nn.Linear(60, 64),  # [batch, 64, 64]
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, 256) ,
            nn.ReLU(),
            nn.Linear(256, 512) ,
            nn.ReLU(),
               
        )
        self.residual_expand = nn.Sequential(*[nn.Sequential(ResidualBlock(512), nn.ReLU()) for _ in range(2)])
        # Second part: Reducing from [512, 512] to [3, 1]
        self.reduce = nn.Sequential(
            nn.Linear(512, 256) ,
            nn.ReLU(),
            nn.Linear(256, 128) ,
            nn.ReLU(),
            nn.Linear(128, 64),  
            nn.ReLU(),
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 16),  
            nn.ReLU(), 
            nn.Linear(16, 8), 
            nn.ReLU(),   
            nn.Linear(8, 3),   
            nn.ReLU(),
            LambdaLayer(lambda x: x.permute(0, 2, 1)), 
            nn.Linear(512, 256) ,
            nn.ReLU(),
            nn.Linear(256, 128) ,
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  
            nn.ReLU(),
            nn.Linear(32, 16),  
            nn.ReLU(), 
            nn.Linear(16, 8),   
            nn.ReLU(),
            nn.Linear(8, 1),  
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.residual_expand(x)
        x = self.reduce(x)
        return x.view(x.size(0), 3, 1)
    
#model = Mymodel()
#input_tensor = torch.randn(16, 60, 27)  # 示例输入
#output = model(input_tensor)
#print(output.shape)  # 应输出 [16, 3, 1]