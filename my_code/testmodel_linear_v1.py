import torch
import torch.nn as nn

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # First part: Expanding 60x27 to 256x256
        self.fc1 = nn.Linear(27, 64)  # [batch, 60, 64]
        self.fc2 = nn.Linear(64, 128)  # [batch, 60, 128]
        self.fc3 = nn.Linear(128, 256)  # [batch, 60, 256]
        self.fc4 = nn.Linear(60, 128)  # [batch, 256, 128]
        self.fc5 = nn.Linear(128, 256)  # [batch, 256, 256]
        self.flatten = nn.Flatten()  # [batch, 65536]

        # Second part: Reducing 65536 to 3x1
        self.fc_reduce1 = nn.Linear(65536, 1024)  # Step down to 1024
        self.fc_reduce2 = nn.Linear(1024, 512)  # Step down to 512
        self.fc_reduce3 = nn.Linear(512, 256)   # Step down to 256
        self.fc_reduce4 = nn.Linear(256, 128)   # Step down to 128
        self.fc_reduce5 = nn.Linear(128, 64)     # Step down to 64
        self.fc_reduce6 = nn.Linear(64, 32)     # Step down to 32
        self.fc_reduce7 = nn.Linear(32, 16)     # Step down to 16
        self.fc_reduce8 = nn.Linear(16, 8)     # Step down to 8
        self.fc_reduce9 = nn.Linear(8, 3)       # Step down to 3x1

    def permute(self, x):
        # 维度置换方法
        return x.permute(0, 2, 1)
    
    def forward(self, x):
        # First part processing
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.permute(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.flatten(x)

        # Second part processing
        x = self.fc_reduce1(x)
        x = torch.relu(x)
        x = self.fc_reduce2(x)
        x = torch.relu(x)
        x = self.fc_reduce3(x)
        x = torch.relu(x)
        x = self.fc_reduce4(x)
        x = torch.relu(x)
        x = self.fc_reduce5(x)
        x = torch.relu(x)
        x = self.fc_reduce6(x)
        x = torch.relu(x)
        x = self.fc_reduce7(x)
        x = torch.relu(x)
        x = self.fc_reduce8(x)         
        x = torch.relu(x)
        x = self.fc_reduce9(x)                
        x = x.view(x.size(0), 3, 1)  # Reshape to [batchsize, 3, 1]

        return x
