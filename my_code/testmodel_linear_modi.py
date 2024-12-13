import torch
import torch.nn as nn

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # First part: Expanding 60x27 to 64x64
        self.expand = nn.Sequential(
            nn.Linear(27, 64),  # [batch, 60, 64]
            nn.ReLU(),
            LambdaLayer(lambda x: x.permute(0, 2, 1)),  # Permute to [batch, 64, 60]
            nn.Linear(60, 64),  # [batch, 64, 64]
            nn.ReLU()
        )
        
        # Replace residual blocks with 10 FC layers
        self.fc_layers = nn.Sequential(*[nn.Sequential(nn.Linear(64, 64), nn.ReLU()) for _ in range(10)])
        
        # Second part: Reducing 4096 to 3x1
        self.reduce = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.fc_layers(x)
        x = self.reduce(x)
        return x.view(x.size(0), 3, 1)