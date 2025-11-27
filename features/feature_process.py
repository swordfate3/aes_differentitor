import numpy as np
from scipy.stats import ks_2samp
from .stat_tests import STAT_TEST_FUNCTIONS
from utils.common_utils import load_config_yaml

# 新增代码：移除对不存在的 utils.constants 依赖，改为从 YAML 加载配置
_cfg = load_config_yaml()
BATCH_NUM = int(_cfg.get("BATCH_NUM", 1))  # 新增代码：批次数量，默认1
ALPHA = float(_cfg.get("ALPHA", 0.05))  # 新增代码：显著性水平，默认0.05
STAT_TESTS = list(_cfg.get("STAT_TESTS", []))  # 新增代码：特征名称列表


def extract_batch_features(batch, ref_batch):
    """提取单个批次的统计特征

    使用 `STAT_TEST_FUNCTIONS` 依次计算特征，其中部分测试需要参考批次。

    Args:
        batch (list | np.ndarray): 当前批次样本（128bit整数列表）
        ref_batch (list | np.ndarray): 参考随机批次（用于双样本对比）

    Returns:
        np.ndarray: 特征向量，长度与 `STAT_TEST_FUNCTIONS` 相同

    Example:
        >>> import secrets
        >>> b = [secrets.randbits(128) for _ in range(16)]
        >>> r = [secrets.randbits(128) for _ in range(16)]
        >>> feats = extract_batch_features(b, r)
        >>> feats.shape[0] == len(STAT_TEST_FUNCTIONS)
        True
    """
    features = []
    for i, test_func in enumerate(STAT_TEST_FUNCTIONS):
        # 需参考随机批次的测试（第7、10、16、20个测试）
        if i in [6, 9, 15, 19]:  # f_test、ks_test、t_test、mann_whitney
            feat_val = test_func(batch, ref_batch)
        else:
            feat_val = test_func(batch)
        features.append(feat_val)
    return np.array(features)


def extract_all_features(cipher_batches, random_datasets):
    """提取所有批次的特征（密文+随机数据）

    从密文批次与随机批次中，分别计算每个批次的特征向量。

    Args:
        cipher_batches (list): 密文批次列表，长度为 `BATCH_NUM`
        random_datasets (list): 随机数据集列表，至少包含两个集合

    Returns:
        tuple[np.ndarray, np.ndarray]: `(cipher_features, random_features)`

    Raises:
        IndexError: 当 `random_datasets` 不满足要求时抛出

    Example:
        >>> # 见项目数据生成模块，提供真实批次后使用
    """
    # 修复代码：参考批次与评估批次解耦，使用同一随机集合的不同批次避免泄漏
    total_batches = min(len(cipher_batches), len(random_datasets))
    ref_batches = [random_datasets[i] for i in range(total_batches)]
    rand_for_feats = [random_datasets[(i + 1) % total_batches] for i in range(total_batches)]
    cipher_features = []
    random_features = []

    # 提取密文特征
    print("提取AES密文特征...")
    for i in range(min(BATCH_NUM, len(cipher_batches), len(ref_batches))):
        cipher_batch = cipher_batches[i]
        ref_batch = ref_batches[i]
        feat = extract_batch_features(cipher_batch, ref_batch)
        cipher_features.append(feat)

    # 提取随机数据特征（第二个随机数据集，避免数据泄露）
    print("提取随机数据特征...")
    for i in range(min(BATCH_NUM, len(rand_for_feats), len(ref_batches))):
        random_batch = rand_for_feats[i]
        ref_batch = ref_batches[i]
        feat = extract_batch_features(random_batch, ref_batch)
        random_features.append(feat)

    cf = np.array(cipher_features, dtype=np.float64)
    rf = np.array(random_features, dtype=np.float64)
    # 新增代码：清理NaN/Inf，防止模型训练失败
    cf = np.nan_to_num(cf, nan=0.0, posinf=1e6, neginf=-1e6)
    rf = np.nan_to_num(rf, nan=0.0, posinf=1e6, neginf=-1e6)
    return cf, rf


def filter_features(cipher_features, random_features):
    """用 KS 双样本测试筛选有效特征

    对每个特征维度进行 KS 测试，p 值小于 `ALPHA` 的特征被保留。

    Args:
        cipher_features (np.ndarray): 密文特征矩阵，形状 `(BATCH_NUM, F)`
        random_features (np.ndarray): 随机特征矩阵，形状 `(BATCH_NUM, F)`

    Returns:
        list[int]: 保留特征的索引列表

    Example:
        >>> import numpy as np
        >>> cf = np.random.randn(4, 3)
        >>> rf = np.random.randn(4, 3)
        >>> idx = filter_features(cf, rf)
        >>> isinstance(idx, list)
        True
    """
    valid_indices = []
    print("筛选有效特征...")
    ks_stats = []
    for i in range(cipher_features.shape[1]):
        cipher_feat = cipher_features[:, i]
        random_feat = random_features[:, i]
        stat, p_val = ks_2samp(cipher_feat, random_feat)
        ks_stats.append((i, float(stat), float(p_val)))
        if p_val < ALPHA:
            valid_indices.append(i)

    # 输出筛选结果
    print(f"筛选结果：{len(valid_indices)}/{cipher_features.shape[1]} 个特征保留")
    if not valid_indices:
        # 新增代码：当无特征通过显著性阈值，回退为按KS统计量选Top半数特征
        ks_sorted = sorted(ks_stats, key=lambda x: x[1], reverse=True)
        top_k = max(1, cipher_features.shape[1] // 2)
        valid_indices = [idx for idx, _, _ in ks_sorted[:top_k]]
        print(f"显著性筛选为空，回退选择Top{top_k} KS统计量特征")
    if STAT_TESTS:
        print(f"保留特征名称：{[STAT_TESTS[i] for i in valid_indices]}")
    return valid_indices


def prepare_train_data(cipher_features, random_features, valid_indices=None):
    """准备训练/测试数据（标签+数据划分）

    Args:
        cipher_features (np.ndarray): 密文特征矩阵
        random_features (np.ndarray): 随机特征矩阵
        valid_indices (list[int], optional): 已筛选的有效特征索引。若为空，默认使用所有特征。

    Returns:
        tuple[np.ndarray, np.ndarray]: `(X, y)` 合并后的数据与标签

    Example:
        >>> import numpy as np
        >>> cf = np.random.randn(4, 3)
        >>> rf = np.random.randn(4, 3)
        >>> X, y = prepare_train_data(cf, rf, [0, 2])
        >>> X.shape[0] == y.shape[0]
        True
    """
    # 生成标签（1=密文，0=随机）
    if valid_indices is None:
        X_cipher = cipher_features
        X_random = random_features
    else:
        X_cipher = cipher_features[:, valid_indices]
        X_random = random_features[:, valid_indices]
    y_cipher = np.ones(X_cipher.shape[0])
    y_random = np.zeros(X_random.shape[0])

    # 合并数据
    X = np.vstack([X_cipher, X_random])
    # 新增代码：再次清理潜在NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.hstack([y_cipher, y_random])

    return X, y
