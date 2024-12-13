import torch.nn as nn
import torch

class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=2),  
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=2),
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
    ([1, 1, 60,27])
    下采样：
    e1: torch.Size([1, 64, 58, 25])
    e2: torch.Size([1, 128, 27, 10])
    e3: torch.Size([1, 256, 11, 3])

"""

class Mymodel(nn.Module):

    def __init__(self, in_ch=1, out_ch=2):
        super(Mymodel, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4] #64,128,256
        out_size = [(24,26), (56,60), (3,8)]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        # 第一层卷积，尺寸缩小，通道数减半
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8448, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        e1 = self.Conv1(x)
        # print('e1:',e1.shape)

        e2 = self.Maxpool1(e1)
        # print('e2:',e2.shape)
        e2 = self.Conv2(e2)
        # print('e2:',e2.shape)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        # print('e3:',e3.shape)
        
        x = self.flatten(e3)  # Flatten the [1, 256, 11, 3] to [1, 8448]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
       
      
     


        out = torch.squeeze(x,1)
        out = torch.unsqueeze(out,2)
        

        # del e1,e2,e3,e4,d1,d2,d3
        # torch.cuda.empty_cache()

        return out





    

# before pre: torch.Size([1, 1, 91, 109, 91])当前输入
# img = torch.rand((1, 1, 512, 3, 6, 3))
# upsample1 = nn.Upsample(size=(15,20,15))
# out1 = upsample1(img)
# print(out1.shape)

#img = torch.rand((64,60, 27))
#my_unet = Mymodel()
#out = my_unet(img)
#print(out.shape)

