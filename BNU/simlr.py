#!/usr/bin/env python3
import numpy as np
import math
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS = np.finfo(float).eps

# 1. 计算平方欧氏距离
def squared_euclidean_distance(X, Y=None):
    """
    计算矩阵 X 与 Y 中各行之间的平方欧氏距离。
    若 Y 为 None，则返回 X 自身的距离矩阵。
    
    X: (M, d)
    Y: (N, d)
    返回: (M, N) 距离矩阵
    """
    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("维度不匹配")
    X2 = np.sum(X**2, axis=1, keepdims=True)  # (M,1)
    Y2 = np.sum(Y**2, axis=1, keepdims=True)  # (N,1)
    dist = X2 + Y2.T - 2 * (X @ Y.T)
    dist[dist < 0] = 0
    return dist

# 2. 根据每行选取最大 k 个值，并对称化
def knn_symmetric_matrix(sim_matrix, k):
    """
    对相似矩阵 sim_matrix 每行保留 k 个最大值，然后构造对称矩阵。
    
    sim_matrix: (N, N)
    k: int, 邻居数
    返回: 对称化后的矩阵、每行的 top-k 值、对应的索引
    """
    sorted_vals = np.sort(sim_matrix, axis=1)[:, ::-1]
    sorted_idx = np.argsort(sim_matrix, axis=1)[:, ::-1]
    top_vals = sorted_vals[:, :k]
    top_idx = sorted_idx[:, :k]
    knn_mat = np.zeros_like(sim_matrix)
    rows = np.arange(sim_matrix.shape[0])[:, None]
    knn_mat[rows, top_idx] = top_vals
    sym_mat = (knn_mat + knn_mat.T) / 2.0
    return sym_mat, top_vals, top_idx

# 3. 特征分解，取前 num_components 个特征向量
def eigen_decomposition(matrix, num_components, largest=True, make_symmetric=True):
    """
    对 matrix 进行特征分解，并返回前 num_components 个特征向量及对应特征值。
    
    matrix: (N, N)
    num_components: int
    largest: 若 True，则返回最大的；否则返回最小的。
    make_symmetric: 是否先对称化
    """
    if make_symmetric:
        matrix = (matrix + matrix.T) / 2.0
    vals, vecs = eigh(matrix)
    if largest:
        idx = np.argsort(vals)[::-1]
    else:
        idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx].real
    return vecs[:, :num_components], vals[:num_components], vals

# 4. 根据核权重融合多个距离核
def weighted_kernel_distance(kernel_matrices, kernel_weights):
    """
    kernel_matrices: (N, N, m)
    kernel_weights: (m,)
    返回加权融合后的 (N, N) 距离矩阵
    """
    return np.tensordot(kernel_matrices, kernel_weights, axes=([2], [0]))

# 5. 计算 L2 距离（假定每行是一个样本）
def l2_distance(X, Y):
    """
    计算 X 与 Y 各行间的欧氏距离（非平方）
    X: (M, d)
    Y: (N, d)
    返回: (M, N)
    """
    dist_sq = squared_euclidean_distance(X, Y)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 0)
    return dist

# 6. 轻量级 K-means 实现（替换原代码）
def lite_kmeans(X, num_clusters, max_iter=100, n_init=1, **kwargs):
    """
    简单实现 K-means，使用平方欧氏距离，支持 'distance' 和 'start' 可选参数。
    
    参数：
      X: (N, d) 数据矩阵
      num_clusters: int, 聚类数
      max_iter: int, 最大迭代次数
      n_init: int, 重复初始化次数
    
    可选参数（kwargs）：
      distance: 'sqeuclidean' 或 'cosine'（默认 'sqeuclidean'）
      start: 'sample'、'cluster' 或 numeric (初始中心或索引)，默认 'sample'
      clustermaxiter: int, 当 start='cluster' 时的预聚类最大迭代次数（默认10）
    
    返回：
      best_labels: (N,) 聚类标签（从0开始，本函数最后输出时加1使其与 MATLAB 相符）
      best_centers: (num_clusters, d) 聚类中心
      converged: bool, 是否在 max_iter 内收敛
      best_sum: float, 聚类目标函数值（簇内距离总和）
      None (占位)
    """
    # 解析可选参数
    distance = kwargs.get('distance', 'sqeuclidean').lower()
    start_method = kwargs.get('start', 'sample')
    clustermaxiter = kwargs.get('clustermaxiter', 10)
    
    N, d = X.shape
    best_sum = math.inf
    best_labels = None
    best_centers = None
    converged = False
    for _ in range(n_init):
        if isinstance(start_method, str):
            if start_method.lower() == 'sample':
                init_idx = np.random.choice(N, num_clusters, replace=False)
                centers = X[init_idx, :].copy()
            elif start_method.lower() == 'cluster':
                sub_size = max(int(math.floor(0.1 * N)), 1)
                sub_idx = np.random.choice(N, sub_size, replace=False)
                X_sub = X[sub_idx, :]
                # 预聚类，调用 lite_kmeans 本身（强制 'sample' 初始化，单次运行，最大迭代 clustermaxiter）
                _, centers = lite_kmeans(X_sub, num_clusters, max_iter=clustermaxiter, n_init=1, start='sample')
            else:
                raise ValueError(f"Unknown start method: {start_method}")
        else:
            # 若 start_method 为 numeric 数组
            arr = np.array(start_method)
            if arr.ndim == 1:
                init_idx = arr[:num_clusters].astype(int)
                centers = X[init_idx, :].copy()
            elif arr.ndim == 2:
                if arr.shape != (num_clusters, d):
                    raise ValueError("Numeric start array must have shape (k, d)")
                centers = arr.copy()
            else:
                raise ValueError("Invalid numeric start parameter")
        
        labels = np.zeros(N, dtype=int)
        prev_labels = -np.ones(N, dtype=int)
        for it in range(max_iter):
            if distance == 'sqeuclidean':
                dist_mat = squared_euclidean_distance(X, centers)
            elif distance == 'cosine':
                # cosine: 1 - (x dot c) / (||x|| ||c||)
                norm_X = np.linalg.norm(X, axis=1, keepdims=True)
                norm_centers = np.linalg.norm(centers, axis=1, keepdims=True)
                dot = X @ centers.T
                # 防止除零
                norm_X[norm_X==0] = EPS
                norm_centers[norm_centers==0] = EPS
                sim = dot / (norm_X @ norm_centers.T)
                dist_mat = 1 - sim
            else:
                raise ValueError("Unknown distance type.")
            new_labels = np.argmin(dist_mat, axis=1)
            # 空簇处理：若有簇空了，则将一些距离较远的样本重新分配
            unique_labels = np.unique(new_labels)
            if len(unique_labels) < num_clusters:
                missing = list(set(range(num_clusters)) - set(unique_labels))
                # 选取距离最远的样本
                overall_dist = np.sum(dist_mat, axis=1)
                sorted_idx = np.argsort(overall_dist)[::-1]
                for m, sample_idx in enumerate(sorted_idx[:len(missing)]):
                    new_labels[sample_idx] = missing[m]
            if np.all(new_labels == prev_labels):
                converged = True
                break
            prev_labels = new_labels.copy()
            # 更新中心
            for j in range(num_clusters):
                if np.any(new_labels == j):
                    centers[j, :] = np.mean(X[new_labels == j, :], axis=0)
            labels = new_labels
        # 计算目标函数值：簇内距离之和
        if distance == 'sqeuclidean':
            dist_mat = np.sqrt(squared_euclidean_distance(X, centers))
        else:
            dist_mat = l2_distance(X, centers)
        sum_d = np.array([np.sum(dist_mat[labels == j, j]) for j in range(num_clusters)])
        total = np.sum(sum_d)
        if total < best_sum:
            best_sum = total
            best_labels = labels.copy()
            best_centers = centers.copy()
    # MATLAB 版本输出 label 从1开始
    return best_labels, best_centers, converged, best_sum, None

# 7. 矩阵归一化（两种模式）
def normalize_matrix(W, mode='average'):
    """
    对矩阵 W 进行归一化：
      mode='average': 每行除以行和
      mode='graph': 每行除以行和的平方根（对称归一化）
    """
    W = W * W.shape[0]
    row_sum = np.sum(np.abs(W), axis=1) + EPS
    if mode == 'average':
        return (1.0 / row_sum)[:, None] * W
    elif mode == 'graph':
        inv_sqrt = 1.0 / np.sqrt(row_sum)
        return inv_sqrt[:, None] * W * inv_sqrt[None, :]
    else:
        return W

# 8. 将每行向量投影到概率单纯形（所有元素非负，和为1）
def project_onto_simplex(X):
    """
    对矩阵 X 的每一行投影到概率单纯形上
    参考算法: Duchi 等
    """
    m, n = X.shape
    Y = np.zeros_like(X)
    for i in range(m):
        v = X[i, :]
        u = np.sort(v)[::-1]
        sv = np.cumsum(u)
        rho_arr = np.nonzero(u * np.arange(1, n+1) > (sv - 1))[0]
        if len(rho_arr) == 0:
            theta = (sv[-1] - 1) / n
        else:
            rho = rho_arr[-1]
            theta = (sv[rho] - 1) / (rho + 1)
        Y[i, :] = np.maximum(v - theta, 0)
    return Y

# 9. 核权重更新（利用二分搜索求解 Hbeta，使得熵接近目标）
def update_kernel_weights(D_vec, beta_val=None):
    """
    D_vec: (m,) 每个核的某种统计量（例如均值）
    beta_val: 初始 beta 参数，若 None 则取 1/m
    返回: 权重向量 P (m,)（归一化后）
    """
    m = len(D_vec)
    if beta_val is None:
        beta_val = 1.0 / m
    tol = 1e-4
    target_H = np.log(20.0)  # 目标 perplexity = 20
    beta_curr = beta_val
    H, P = compute_Hbeta(D_vec, beta_curr)
    Hdiff = H - target_H
    betamin = -np.inf
    betamax = np.inf
    tries = 0
    while abs(Hdiff) > tol and tries < 30:
        if Hdiff > 0:
            betamin = beta_curr
            beta_curr = beta_curr * 2 if np.isinf(betamax) else (beta_curr + betamax) / 2.0
        else:
            betamax = beta_curr
            beta_curr = beta_curr / 2 if np.isinf(betamin) else (beta_curr + betamin) / 2.0
        H, P = compute_Hbeta(D_vec, beta_curr)
        Hdiff = H - target_H
        tries += 1
    return P

def compute_Hbeta(D, beta):
    """
    计算 H 和概率向量 P，其中 D 先归一化到 [0,1]。
    D: 数值或向量
    beta: 标量
    返回: H, P
    """
    D_norm = (D - np.min(D)) / (np.max(D) - np.min(D) + EPS)
    P = np.exp(-D_norm * beta)
    sumP = np.sum(P) + EPS
    H = np.log(sumP) + beta * np.sum(D_norm * P) / sumP
    P = P / sumP
    return H, P

# 10. 多核构造：利用不同带宽的高斯核生成多个距离矩阵
def multiple_kernels(X):
    """
    根据输入数据 X 构造多个高斯核（这里只构造距离矩阵）。
    
    返回: 一个 3D 数组，形状 (N, N, m) ，m 为核的个数。
    """
    N = X.shape[0]
    sigma_values = np.arange(2, 0.75, -0.25)  # 2, 1.75, 1.5, 1.25, 1.0
    Diff = squared_euclidean_distance(X)  # (N, N)
    sorted_Diff = np.sort(Diff, axis=1)
    all_k = np.arange(10, 32, 2)  # 10, 12, ... , 30
    kernel_list = []
    for k_val in all_k:
        if k_val < (N - 1):
            TT = np.mean(sorted_Diff[:, 1:(k_val+1)], axis=1) + EPS
            Sig = (np.tile(TT, (N, 1)).T + np.tile(TT, (N, 1))) / 2.0
            Sig = np.maximum(Sig, EPS)
            for sigma in sigma_values:
                W = norm.pdf(Diff, 0, sigma * Sig)
                W = (W + W.T) / 2.0
                kernel_list.append(W)
    kernels = np.stack(kernel_list, axis=2)  # (N, N, m)
    D_kernels = np.zeros_like(kernels)
    for i in range(kernels.shape[2]):
        K = kernels[:, :, i]
        diagK = np.diag(K)
        D_i = np.add.outer(diagK, diagK) - 2 * K
        np.fill_diagonal(D_i, 0)
        D_kernels[:, :, i] = D_i
    return D_kernels

# 11. 网络扩散：利用局部邻域信息平滑相似矩阵
def network_diffusion(sim_matrix, k):
    """
    sim_matrix: (N, N) 初始相似矩阵（如 S0 = max(fused_dist)-fused_dist）
    k: 邻居数
    返回: 平滑后的相似矩阵
    """
    sim_no_diag = sim_matrix - np.diag(np.diag(sim_matrix))
    P, _, _ = knn_symmetric_matrix(np.abs(sim_no_diag), min(k, sim_matrix.shape[0]-1))
    P = P * np.sign(sim_matrix)
    D_vector = np.sum(np.abs(P), axis=1)
    P_plus = P + np.eye(P.shape[0]) + np.diag(np.sum(np.abs(P), axis=1))
    P_trans = transition_fields(P_plus)
    U, D_vals, _ = eigen_decomposition(P_trans, P_trans.shape[0], largest=True)
    d = np.real(D_vals) + EPS
    alpha_val = 0.8
    beta_val = 2.0
    d_new = (1 - alpha_val) * d / (1 - alpha_val * (d ** beta_val))
    D_mat = np.diag(d_new)
    W = U @ D_mat @ U.T
    diagW = np.diag(W)
    W = (W - np.diag(diagW)) / (1 - diagW + EPS)
    W = np.diag(D_vector) @ W
    W = (W + W.T) / 2.0
    return W

# 12. t-SNE 降维
def tsne_embedding(P, no_dims=2, labels=None):
    """
    Performs symmetric t-SNE on an affinity matrix P.
    
    This function is a direct Python translation of the MATLAB function tsne_p_bo.
    It uses an iterative optimization to minimize the KL divergence between P and Q
    (computed using the Student-t distribution), with adaptive learning rates.
    
    Parameters
    ----------
    P : ndarray, shape (n, n)
        Affinity matrix (should be symmetric, with diagonal 0 and normalized to sum=1).
    no_dims : int or ndarray, optional
        If an int, the target dimensionality (default=2).
        If an array is provided, it is interpreted as an initial embedding solution,
        and no_dims is then set to its number of columns.
    labels : array-like, optional
        Optional labels (not used in the optimization).
    
    Returns
    -------
    Y : ndarray, shape (n, no_dims)
        The final t-SNE embedding.
    """
    # If labels not provided, set to empty array
    if labels is None:
        labels = np.array([])
    
    # Check if an initial solution is provided (if no_dims is an array)
    if isinstance(no_dims, (list, np.ndarray)):
        initial_solution = True
        Y = np.array(no_dims, dtype=float).copy()
        no_dims = Y.shape[1]
    else:
        initial_solution = False
        no_dims = int(no_dims)
    
    n = P.shape[0]
    momentum = 0.08
    final_momentum = 0.1
    momentum_switch_iter = 250
    stop_lying_iter = 100
    max_iter = 1000
    learning_rate = 500
    min_gain = 0.01

    # Ensure P has zero diagonal and is symmetric
    np.fill_diagonal(P, 0)
    P = 0.5 * (P + P.T)
    total_P = np.sum(P)
    P = np.maximum(P / total_P, np.finfo(float).tiny)
    
    # Constant in KL divergence (unused later)
    const_KL = np.sum(P * np.log(P))
    
    if not initial_solution:
        # Lie about P-values to get better local minima
        P = P * 4
    
    if not initial_solution:
        # Random initialization for embedding
        Y = 0.0001 * np.random.randn(n, no_dims)
    
    Y_increments = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    # Main t-SNE optimization loop
    for iter_idx in range(max_iter):
        # Compute squared Euclidean distances in the embedding space
        sum_Y = np.sum(Y**2, axis=1, keepdims=True)
        dist_Y = sum_Y + sum_Y.T - 2 * (Y @ Y.T)
        
        # Compute Student-t based joint probabilities Q
        num = 1.0 / (1 + dist_Y)
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, np.finfo(float).tiny)
        
        # Compute gradient: gradient = 4 * (diag(sum(L,1)) - L) * Y, where L = (P - Q) .* num
        L_matrix = (P - Q) * num
        sum_L = np.sum(L_matrix, axis=0)
        grad_Y = 4 * ((np.diag(sum_L) - L_matrix) @ Y)
        
        # Update gains: increase gain when gradient changes sign, otherwise decay
        sign_mismatch = np.not_equal(np.sign(grad_Y), np.sign(Y_increments))
        gains = (gains + 0.2) * (~sign_mismatch) + (gains * 0.8) * sign_mismatch
        gains[gains < min_gain] = min_gain
        
        # Update increments and embedding
        Y_increments = momentum * Y_increments - learning_rate * (gains * grad_Y)
        Y = Y + Y_increments
        Y = Y - np.mean(Y, axis=0)
        Y = np.clip(Y, -100, 100)
        
        # Update momentum if needed
        if iter_idx == momentum_switch_iter:
            momentum = final_momentum
        if iter_idx == stop_lying_iter and not initial_solution:
            P = P / 4
    
    return Y

# 13. TransitionFields：对矩阵进行归一化与重构
def transition_fields(W):
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    W = W * W.shape[0]
    W = normalize_matrix(W, mode='average')
    col_norm = np.sqrt(np.sum(np.abs(W), axis=0)) + EPS
    W = W / np.tile(col_norm, (W.shape[0], 1))
    W = W @ W.T
    W_new = W.copy()
    if zero_idx.size > 0:
        W_new[zero_idx, :] = 0
        W_new[:, zero_idx] = 0
    return W_new

# 14. SIMLR 主流程
def simlr_cluster(X, num_clusters):
    """
    SIMLR 聚类算法主函数。
    参数:
      num_clusters: 要分的簇数
      X: (n_samples, n_features) 数据矩阵
    返回:
      labels: (n_samples,) 聚类标签
    """
    n_samples = X.shape[0]
    k_neighbors = 30
    num_iterations = 30
    beta = 0.8
    r_value = -1

    D_kernels = multiple_kernels(X)  # (n_samples, n_samples, num_kernels)
    num_kernels = D_kernels.shape[2]
    kernel_weights = np.ones(num_kernels) / num_kernels
    fused_dist = np.mean(D_kernels, axis=2)  # (n_samples, n_samples)

    sorted_indices = np.argsort(fused_dist, axis=1)
    sorted_dists = np.sort(fused_dist, axis=1)

    A = np.zeros((n_samples, n_samples))
    d_neighbors = sorted_dists[:, 1:(k_neighbors+2)]
    r_vec = 0.5 * (k_neighbors * d_neighbors[:, k_neighbors] - np.sum(d_neighbors[:, :k_neighbors], axis=1))
    neighbor_idx = sorted_indices[:, 1:(k_neighbors+2)]
    d_last = d_neighbors[:, k_neighbors:k_neighbors+1]
    denom = k_neighbors * d_last - np.sum(d_neighbors[:, :k_neighbors], axis=1, keepdims=True) + EPS
    weights = (d_last - d_neighbors) / denom
    for i in range(n_samples):
        A[i, neighbor_idx[i, :]] = weights[i, :]
    if r_value <= 0:
        r_value = np.mean(r_vec)
    lambda_param = max(np.mean(r_vec), 0)
    A[np.isnan(A)] = 0

    S0 = np.max(fused_dist) - fused_dist
    S0 = network_diffusion(S0, k_neighbors)
    S0 = normalize_matrix(S0, mode='average')
    S = (S0 + S0.T) / 2.0
    D_mat = np.diag(np.sum(S, axis=1))
    L0 = D_mat - S
    F, _, _ = eigen_decomposition(L0, num_clusters, largest=False)
    F = normalize_matrix(F, mode='average')

    converge_history = []
    S_old = S.copy()

    for iter_idx in range(num_iterations):
        dist_F = l2_distance(F, F)
        A_new = np.zeros((n_samples, n_samples))
        neighbor_sorted = sorted_indices[:, 1:]
        for i in range(n_samples):
            indices = neighbor_sorted[i, :]
            tmp = (fused_dist[i, indices] + lambda_param * dist_F[i, indices]) / (2 * r_value)
            proj = project_onto_simplex(-tmp.reshape(1, -1)).flatten()
            A_new[i, indices] = proj
        A_new[np.isnan(A_new)] = 0
        S = (1 - beta) * A_new + beta * S
        S = network_diffusion(S, k_neighbors)
        S = (S + S.T) / 2.0

        D_new = np.diag(np.sum(S, axis=1))
        L_new = D_new - S
        F_old = F.copy()
        F, _, ev = eigen_decomposition(L_new, num_clusters, largest=False)
        F = normalize_matrix(F, mode='average')
        F = (1 - beta) * F_old + beta * F

        DD = np.zeros(num_kernels)
        for i in range(num_kernels):
            temp = (EPS + D_kernels[:, :, i]) * (EPS + S)
            DD[i] = np.mean(np.sum(temp, axis=1))
        new_kernel_weights = update_kernel_weights(DD)
        kernel_weights = (1 - beta) * kernel_weights + beta * new_kernel_weights
        kernel_weights /= np.sum(kernel_weights)

        fn1 = np.sum(ev[:num_clusters])
        full_eig = eigh(L_new)[0]
        fn2 = np.sum(full_eig[:num_clusters+1])
        converge_val = fn2 - fn1
        converge_history.append(converge_val)
        if iter_idx > 0 and converge_val > 1.01 * converge_history[iter_idx - 1]:
            S = S_old
            break
        S_old = S.copy()
        fused_dist = weighted_kernel_distance(D_kernels, kernel_weights)
        sorted_indices = np.argsort(fused_dist, axis=1)

    LF = F.copy()
    tsne_result = tsne_embedding(S, no_dims=num_clusters)
    labels, _, _, _, _ = lite_kmeans(tsne_result, num_clusters, max_iter=100, n_init=20)
    return labels

# 15. 示例调用
if __name__ == '__main__':
    np.random.seed(42)
    # 生成测试数据：100个样本，10个特征
    X = np.random.rand(100, 10)
    num_clusters = 3
    cluster_labels = simlr_cluster(num_clusters, X)
    print("SIMLR 聚类结果标签：", cluster_labels)
