import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from utils.common_utils import load_config_yaml

# 加载配置
config = load_config_yaml()


def create_plot_dir():
    """创建可视化图保存目录"""
    os.makedirs(config["plot"]["save_dir"], exist_ok=True)
    print(f"可视化图将保存到：{config['plot']['save_dir']}")


def plot_pca_features(X_cipher, X_random, valid_indices):
    """PCA降维可视化（特征分布聚类）

    当特征维度不足2时（例如仅保留1维），自动退化为将该维度复制为两列，以
    满足PCA的二维输出需求；同时在样本数非常少时保持鲁棒输出。
    """
    create_plot_dir()
    # 合并数据
    X = np.vstack([X_cipher, X_random])
    y = np.hstack([np.ones(X_cipher.shape[0]), np.zeros(X_random.shape[0])])
    # 处理单维场景：复制一列以避免 sklearn 对 n_components=2 的约束
    if X.shape[1] < 2:
        X = np.hstack([X, X])
    # PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        c="red",
        label="AES Ciphertext",
        alpha=0.6,
        s=50,
    )
    plt.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        c="blue",
        label="Random Data",
        alpha=0.6,
        s=50,
    )
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.title("PCA of Statistical Features (Ciphertext vs Random Data)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["plot"]["save_dir"], "pca_features.png"), dpi=300)
    plt.close()
    print("PCA可视化图保存完成")


def plot_tsne_features(X_cipher, X_random, valid_indices):
    """tSNE降维可视化（特征分布聚类）

    通过tSNE将统计特征嵌入到二维空间，展示密文与随机数据的分离情况。

    Args:
        X_cipher (np.ndarray): 密文特征矩阵（筛选后）
        X_random (np.ndarray): 随机特征矩阵（筛选后）
        valid_indices (list[int]): 保留的特征索引

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> plot_tsne_features(np.random.randn(10,3), np.random.randn(10,3), [0,1,2])
    """
    create_plot_dir()
    X = np.vstack([X_cipher, X_random])
    y = np.hstack([np.ones(X_cipher.shape[0]), np.zeros(X_random.shape[0])])
    # 新增代码【FIX-ADDED】：标准化与裁剪以避免tSNE内部距离溢出
    # 将每个特征归一化为零均值单位方差，并裁剪到[-10, 10]
    X = X.astype(np.float64)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    X = np.clip(X, -10.0, 10.0)
    # 修改代码：自适应perplexity，确保小样本下有效
    n_samples = X.shape[0]
    perplexity = max(5, min(30, n_samples // 3))
    # 修改代码【FIX-CHANGED】：当特征维度为1时复制列并加微噪声（标准化后）
    if X.shape[1] < 2:
        noise = np.random.normal(0, 1e-6, size=(X.shape[0], 1))
        X = np.hstack([X, X + noise])
    # 修改代码【FIX-CHANGED】：使用自动学习率与PCA初始化提升稳定性
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_emb[y == 1, 0], X_emb[y == 1, 1], c="red", label="AES Ciphertext", alpha=0.6, s=50)
    plt.scatter(X_emb[y == 0, 0], X_emb[y == 0, 1], c="blue", label="Random Data", alpha=0.6, s=50)
    plt.xlabel("tSNE Component 1", fontsize=12)
    plt.ylabel("tSNE Component 2", fontsize=12)
    plt.title("tSNE of Statistical Features (Ciphertext vs Random Data)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["plot"]["save_dir"], "tsne_features.png"), dpi=300)
    plt.close()
    print("tSNE可视化图保存完成")


def plot_top_kde_features(X_cipher, X_random, valid_indices, trained_rf):
    """绘制Top3重要特征的KDE分布对比图"""
    create_plot_dir()
    # 获取随机森林的特征重要性
    importances = trained_rf.feature_importances_
    # 筛选Top3重要特征（相对于筛选后的特征索引）
    top3_idx = np.argsort(importances)[-3:]

    for idx in top3_idx:
        feat_name = config["STAT_TESTS"][valid_indices[idx]]  # 修复代码：配置键名与common.yaml一致，避免KeyError
        # 提取特征值
        cipher_feat = X_cipher[:, idx]
        random_feat = X_random[:, idx]

        # 计算KDE
        kde_cipher = stats.gaussian_kde(cipher_feat)
        kde_random = stats.gaussian_kde(random_feat)
        # 生成x轴范围
        x_min = min(cipher_feat.min(), random_feat.min())
        x_max = max(cipher_feat.max(), random_feat.max())
        x = np.linspace(x_min, x_max, 1000)

        # 绘图
        plt.figure(figsize=(8, 4))
        plt.plot(x, kde_cipher(x), "r-", linewidth=2, label="AES Ciphertext")
        plt.plot(x, kde_random(x), "b-", linewidth=2, label="Random Data")
        plt.xlabel(f"Feature Value ({feat_name})", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(f"KDE Distribution: {feat_name}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config["plot"]["save_dir"], f"kde_{feat_name}.png"), dpi=300)
        plt.close()
    print("Top3特征KDE图保存完成")
