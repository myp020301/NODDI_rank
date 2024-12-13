import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats

class SeqDataset(Dataset):
    def __init__(self, seq_len, nb_sample=400000, dist=None):
        self.seq_len = seq_len
        self.nb_sample = nb_sample
        self.dist = dist

    def __getitem__(self, index):
        rand_seq = get_rand_seq(self.seq_len, self.dist)
        ranks = get_tiedrank(torch.FloatTensor(rand_seq))
        return torch.FloatTensor(rand_seq), ranks

    def __len__(self):
        return self.nb_sample

def get_rand_seq(seq_len, ind=None):
    if ind is None:
        type_rand = np.random.randint(0, 11)
    else:
        type_rand = int(ind)

    if type_rand == 0:
        rand_seq = np.random.rand(seq_len) * 2.0 - 1
    elif type_rand == 1:
        rand_seq = np.random.uniform(-1, 1, seq_len)
    elif type_rand == 2:
        rand_seq = np.random.standard_normal(seq_len)
    elif type_rand == 3:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 4:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
        np.random.shuffle(rand_seq)
    elif type_rand == 5:
        split = np.random.randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.standard_normal(seq_len - split)])
    elif type_rand == 6:
        split = np.random.randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.uniform(-1, 1, split), np.random.standard_normal(seq_len - split)])
    elif type_rand == 7:
        split = np.random.randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.uniform(-1, 1, seq_len - split)])
    elif type_rand == 8:
        split = np.random.randint(1, seq_len)
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / split)
        np.random.shuffle(rand_seq)
        rand_seq = np.concatenate(
            [rand_seq, np.random.rand(seq_len - split) * 2.0 - 1])
    elif type_rand == 9:
        a = -1.0
        b = 1.0
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 10:
        rand_seq = np.zeros(seq_len)
        
    # 标准化到0到1区间
    if np.max(rand_seq) == np.min(rand_seq):
        rand_seq = np.zeros(seq_len)
    else:# 数据无关
        rand_seq = (rand_seq - np.min(rand_seq)) / (np.max(rand_seq) - np.min(rand_seq))

    # 缩放到0到10区间
    rand_seq = rand_seq * 10

    return rand_seq[:seq_len]

def get_tiedrank(batch_score, dim=0):
    batch_score = batch_score.cpu().numpy()
    rank = stats.rankdata(batch_score, method='average')
    rank = (rank - 1) * -1 + batch_score.size
    rank = torch.from_numpy(rank).float()
    rank = rank / batch_score.size
    return rank

## 测试脚本
#if __name__ == "__main__":
#    seq_len = 10  # 序列长度
#    nb_sample = 100  # 样本数量
#    dataset = SeqDataset(seq_len, nb_sample)
#
#    # 创建数据加载器
#    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
#
#    # 打印前几个样本
#    for i, (rand_seq, ranks) in enumerate(dataloader):
#        print(f"Sample {i+1}")
#        print("Random Sequence:")
#        print(rand_seq)
#        print("Ranks:")
#        print(ranks)
#        print("\n")
#        if i >= 1:  # 打印前两个批次
#            break
