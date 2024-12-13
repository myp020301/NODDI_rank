import torch
import torch.nn as nn

class CustomResidualLayer(nn.Module):
    def __init__(self, in_time_steps, in_features, out_time_steps, out_features):
        super(CustomResidualLayer, self).__init__()
        # Transform the features from in_features to out_features
        self.adjust_features = nn.Linear(in_features, out_features)
        # Optional: Transform the time steps if needed
        self.adjust_time_steps = nn.Linear(in_time_steps, out_time_steps)

    def forward(self, x):
        # Adjust features
        x = self.adjust_features(x)
        # Permute to adjust time steps
        x = x.permute(0, 2, 1)
        # Adjust time steps
        x = self.adjust_time_steps(x)
        # Permute back to the original shape [batch, time_step, feature]
        x = x.permute(0, 2, 1)
        return x

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # First part: Expanding 60x27 to 256x256
        self.fc1 = nn.Linear(27, 64)  # [batch, 60, 64]
        self.fc2 = nn.Linear(60, 128)  # [batch, 64, 128]
        self.fc3 = nn.Linear(64, 128)  # [batch, 128, 128]
        self.fc4 = nn.Linear(128, 256)  # [batch, 128, 256]
        self.fc5 = nn.Linear(128, 256)  # [batch, 256, 256]
        self.flatten = nn.Flatten()  # [batch, 65536]
        
        self.residual_transform1 = CustomResidualLayer(60, 27, 64, 60)
        self.residual_transform2 = CustomResidualLayer(64, 60, 128, 64)
        self.residual_transform3 = CustomResidualLayer(128, 64, 128, 128)
        self.residual_transform4 = CustomResidualLayer(128, 128, 256, 128)


        # Second part: Reducing 65536 to 3x1
        self.fc_reduce1 = nn.Linear(65536, 1024)  # Step down to 1024
        self.fc_reduce2 = nn.Linear(1024, 512)  # Step down to 512
        self.fc_reduce3 = nn.Linear(512, 256)   # Step down to 256
        self.fc_reduce4 = nn.Linear(256, 128)   # Step down to 128
        self.fc_reduce5 = nn.Linear(128, 64)    # Step down to 64
        self.fc_reduce6 = nn.Linear(64, 32)     # Step down to 32
        self.fc_reduce7 = nn.Linear(32, 16)     # Step down to 16
        self.fc_reduce8 = nn.Linear(16, 8)      # Step down to 8
        self.fc_reduce9 = nn.Linear(8, 3)       # Step down to 3x1

    def forward(self, x):
        # First part processing
        residual = x
        x1 = self.fc1(x)#[60,64]
        x1 = torch.relu(x1)
        x1 = self.permute(x1) + self.residual_transform1(residual)#[64,60]
        
        residual = x1
        x2 = self.fc2(x1)#[64,128]
        x2 = torch.relu(x2)
        x2 = self.permute(x2) + self.residual_transform2(residual) #[128,64]
        
        
        # Apply adapter and add residual connection
        residual = x2
        x3 = self.fc3(x2)
        x3 = torch.relu(x3)
        x3 = self.permute(x3) + self.residual_transform3(residual)
        
        residual = x3
        x4 = self.fc4(x3)
        x4 = torch.relu(x4)
        x4 = self.permute(x4) + self.residual_transform4(residual)
        

        x5 = self.fc5(x4)
        x5 = torch.relu(x5)
        x5 = self.flatten(x5)

        # Second part processing
        x = self.fc_reduce1(x5)
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
    
    def permute(self, x):
        # 维度置换方法
        return x.permute(0, 2, 1)
    
        
