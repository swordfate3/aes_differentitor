import os
import random
import numpy as np
import torch
import yaml
import secrets


def set_seed(seed: int = 42):
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda") -> torch.device:
    """自动选择可用设备"""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config_yaml(config_path: str = "configs/common.yaml"):
    """加载配置文件（YAML格式）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def generate_random_data(batch_num, sample_per_batch, block_size):
    """生成论文中的随机数据集：每个批次4096个128bit样本"""
    random_batches = []
    for _ in range(batch_num):
        # 每个样本为128bit（16byte），用secrets保证cryptographically secure
        batch = [secrets.randbits(block_size * 8) for _ in range(sample_per_batch)]
        random_batches.append(batch)
    return random_batches
