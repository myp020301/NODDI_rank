import torch.nn as nn
import torch

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3),  
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块    
"""


    
"""
    模型主架构
    ([1, 1, 60,64])
    下采样：
    e1: torch.Size([1, 64, 56, 60])
    e2: torch.Size([1, 128, 24, 26])
    e3: torch.Size([1, 256, 4, 4])
    d5 torch.Size([1, 1, 3,  8])
"""

class Mymodel(nn.Module):

    def __init__(self, in_ch=1, out_ch=2):
        super(Mymodel, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4] #64,128,256

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*4*4, 1024)  # Reduce to 128 features
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 24)
        self.relu = nn.ReLU()

    

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        e1 = self.Conv1(x)#(56,60)
        # print('e1:',e1.shape)

        e2 = self.Maxpool1(e1)#(28,30)
        # print('e2:',e2.shape)
        e2 = self.Conv2(e2)#(24,26)
        # print('e2:',e2.shape)

        e3 = self.Maxpool2(e2)#(12,13)
        e3 = self.Conv3(e3)#(8,9)
        # print('e3:',e3.shape)
        e3 = self.Maxpool2(e3)#(4,4)
        
        x = self.flatten(e3)          # Flatten the [1, 256, 4, 4] to [1, 4096]
        x = self.fc1(x)              # First fully connected layer
        x = self.relu(x)             # ReLU activation
        x = self.fc2(x)              # Second fully connected layer
        x = self.relu(x)             # ReLU activation
        x = self.fc3(x)              # Second fully connected layer
        x = self.relu(x)             # ReLU activation
        x = self.fc4(x)              # Second fully connected layer
        x = self.relu(x)             # ReLU activation
        x = x.view(-1, 1, 3, 8)       # Reshape to [1, 1, 3, 8]
      
        out = torch.squeeze(x,1)

        # del e1,e2,e3,e4,d1,d2,d3
        # torch.cuda.empty_cache()

        return out





    

# before pre: torch.Size([1, 1, 91, 109, 91])当前输入
# img = torch.rand((1, 1, 512, 3, 6, 3))
# upsample1 = nn.Upsample(size=(15,20,15))
# out1 = upsample1(img)
# print(out1.shape)

#img = torch.rand((64,60, 64))
#my_unet = Conv()
#out = my_unet(img)
#print(out.shape)

