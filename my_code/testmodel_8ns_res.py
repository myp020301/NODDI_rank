import torch
import torch.nn as nn

class ThresholdedReLU(nn.Module):
    def __init__(self, theta=0.01):
        super(ThresholdedReLU, self).__init__()
        self.theta = theta

    def forward(self, x):
        return torch.max(x, torch.tensor(self.theta).to(x.device))

class Mymodel(nn.Module):
    def __init__(
        self,
        input_shape=[60, 3 * 3 * 3],
        output_shape=[3, 1 * 1 * 1],
        nDictQ=300,
        nDictS=300,
    ):
        super(Mymodel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nDictQ = nDictQ
        self.nDictS = nDictS

        self.nLayers1 = 8
        self.nLayers2 = 3
        self.ReLUThres = 0.01

        self.nChannelsQ = 75
        self.nChannelsS = 75

        # 定义全连接层
        self.Ws = nn.Linear(self.input_shape[1], self.nDictS)
        self.Wq = nn.Linear(self.input_shape[0], self.nDictQ)
        self.Wfs = nn.Linear(self.input_shape[1], self.nDictS)
        self.Wfq = nn.Linear(self.input_shape[0], self.nDictQ)
        self.Wis = nn.Linear(self.input_shape[1], self.nDictS)
        self.Wiq = nn.Linear(self.input_shape[0], self.nDictQ)

        # 定义 W_fx, W_ix, 和 S 的层
        self.W_fx_layers_S = nn.ModuleList([nn.Linear(self.nDictS, self.nDictS) for _ in range(self.nLayers1 - 1)])
        self.W_fx_layers_Q = nn.ModuleList([nn.Linear(self.nDictQ, self.nDictQ) for _ in range(self.nLayers1 - 1)])
        self.W_ix_layers_S = nn.ModuleList([nn.Linear(self.nDictS, self.nDictS) for _ in range(self.nLayers1 - 1)])
        self.W_ix_layers_Q = nn.ModuleList([nn.Linear(self.nDictQ, self.nDictQ) for _ in range(self.nLayers1 - 1)])
        self.S_layers_S = nn.ModuleList([nn.Linear(self.nDictS, self.nDictS) for _ in range(self.nLayers1 - 1)])
        self.S_layers_Q = nn.ModuleList([nn.Linear(self.nDictQ, self.nDictQ) for _ in range(self.nLayers1 - 1)])
        
        # 初始化 H 模型的层
        self.H_layers = nn.ModuleList()
        self.H_layers.append(nn.Linear(self.nDictS, self.nChannelsS))
        self.H_layers.append(nn.Linear(self.nDictQ, self.nChannelsQ))
        for i in range(self.nLayers2 - 1):
            self.H_layers.append(nn.Linear(self.nChannelsS, self.nChannelsS))
            self.H_layers.append(nn.Linear(self.nChannelsQ, self.nChannelsQ))
        self.H_layers.append(nn.Linear(self.nChannelsS, self.output_shape[1]))
        self.H_layers.append(nn.Linear(self.nChannelsQ, self.output_shape[0]))

    def forward(self, x):
        # 实现前向传播逻辑
        Ws_output = self.permute(self.Ws(x))
        Wq_output = self.permute(self.Wq(Ws_output))
        Wfs_output = self.permute(self.Wfs(x))
        Wfq_output = self.permute(self.Wfq(Wfs_output))
        Wis_output = self.permute(self.Wis(x))
        Wiq_output = self.permute(self.Wiq(Wis_output))

        # ... 后续操作 ...
        Ctilde = Wq_output

        I = torch.sigmoid(Wiq_output)
        C = torch.mul(I, Ctilde)

        Relu = ThresholdedReLU(theta=self.ReLUThres)
        X = Relu(C)
        X_res = X

        for i in range(self.nLayers1 - 1):

            # 在每次迭代中基于更新的 X 计算 W_fx, W_ix, 和 S 的输出
            W_fx_output = self.permute(self.W_fx_layers_Q[i](self.permute(self.W_fx_layers_S[i](X))))
            W_ix_output = self.permute(self.W_ix_layers_Q[i](self.permute(self.W_ix_layers_S[i](X))))
            S_output = self.permute(self.S_layers_Q[i](self.permute(self.S_layers_S[i](X))))

            Diff_X = torch.sub(X, S_output)

            Ctilde = torch.add(Wq_output, Diff_X)

            Wfx_Wfy = torch.add(W_fx_output, Wfq_output)
            F = torch.sigmoid(Wfx_Wfy)
            Wix_Wiy = torch.add(W_ix_output, Wiq_output)
            I = torch.sigmoid(Wix_Wiy)
            Cf = torch.mul(F, C)
            Ci = torch.mul(I, Ctilde)
            C = torch.add(Cf, Ci)
            if i %2 != 0:
                X = Relu(C) + X_res
                X_res = X
            else :
                X = Relu(C)
                

        H_output = X
        for layer in self.H_layers:
            H_output = layer(H_output)  # 先应用层的变换
            H_output = self.permute(H_output)
            H_output = torch.relu(H_output)  # 最后应用激活函数

        return H_output

    def permute(self, x):
        # 维度置换方法
        return x.permute(0, 2, 1)
