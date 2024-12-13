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

        # Gradual reduction in features
        self.fc_reduce1 = nn.Linear(256, 128)  # [batch, 256, 128]
        self.fc_reduce2 = nn.Linear(128, 64)   # [batch, 256, 64]
        self.fc_reduce3 = nn.Linear(64, 32)    # [batch, 256, 32]
        self.fc_reduce4 = nn.Linear(32, 16)    # [batch, 256, 16]
        self.fc_reduce5 = nn.Linear(16, 8)     # [batch, 256, 8]
        self.fc_reduce6 = nn.Linear(8, 3)      # [batch, 256, 3]
        self.fc_reduce7 = nn.Linear(3, 1)      # [batch, 256, 3]

    def forward(self, x):
        # First part processing
        x = torch.relu(self.fc1(x))
        print(x.shape)
        x = torch.relu(self.fc2(x))
        print(x.shape)
        x = torch.relu(self.fc3(x))
        print(x.shape)
        x = x.permute(0, 2, 1)  # [batch, 256, 60]
        x = torch.relu(self.fc4(x))
        print(x.shape)
        x = torch.relu(self.fc5(x))  # [batch, 256, 256]
        print(x.shape)


        x = torch.relu(self.fc_reduce1(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce2(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce3(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce4(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce5(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce6(x))
        print(x.shape)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.fc_reduce1(x))
        print(x.shape)
        x = torch.relu(self.fc_reduce2(x))
        x = torch.relu(self.fc_reduce3(x))
        x = torch.relu(self.fc_reduce4(x))
        x = torch.relu(self.fc_reduce5(x))
        x = torch.relu(self.fc_reduce6(x))
        x = torch.relu(self.fc_reduce7(x))
        print(x.shape)

        return x
    
    def permute(self, x):
        # 维度置换方法
        return x.permute(0, 2, 1)