#!/usr/bin/env python3
import os, yaml
from pathlib import Path
import subprocess, numpy as np, nibabel as nib


# Load & expand env vars
_cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
CFG = {k: os.path.expandvars(v) if isinstance(v, str) else v
       for k, v in _cfg.items()}

# 原始 workdir（不含 atlas_order 子目录）
BASE_WORKDIR = Path(CFG["workdir"])
# 从配置里取出 atlas_order
ATLAS_ORDER = CFG["atlas_order"]

# 最终把 atlas_order 拼接到 workdir 后面
WORKDIR = BASE_WORKDIR / f"Atlas{ATLAS_ORDER}"

# 其它路径定义
DATASET_DIR = Path(CFG["dataset_dir"])

def run(cmd: str):
    print("[CMD]", cmd)
    subprocess.run(cmd, shell=True, check=True)

# 极小常数，防止除零
EPS = 1e-12

def pearson_z(a, b):
    """
    计算两个向量集合的 Pearson 相关并做 Fisher-Z 变换。
    输入：
      a：形状 (N, V) 的 NumPy 数组
      b：形状 (M, V) 的 NumPy 数组
    输出：
      z：形状 (N, M) 的 Fisher-Z 相关矩阵
    """
    # 去均值
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)

    # L2 归一化（每行向量除以其范数 + EPS）
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + EPS)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + EPS)

    # 计算 Pearson 相关系数矩阵 r (N×M)
    r = a @ b.T
    # 限制在 (-0.999999, 0.999999) 以内，避免 log(0) 或 inf
    r = np.clip(r, -0.999999, 0.999999)

    # Fisher-Z 变换
    #z = 0.5 * np.log((1 + r) / (1 - r))
    return r


def overlap_ratio(lbl1: np.ndarray, lbl2: np.ndarray, valid_mask=None):
    """计算两张标签图的重叠 (同值/有效点)。"""
    if valid_mask is None:
        valid_mask = np.ones_like(lbl1, dtype=bool)
    same = (lbl1 == lbl2) & valid_mask
    return same.sum() / valid_mask.sum()
