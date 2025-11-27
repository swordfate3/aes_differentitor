"""密文数据集生成工具

基于 `cipher` 包提供的 AES CBC 加解密，实现批量生成密文数据集。
可从 YAML 配置读取参数，一键生成并保存为 CSV。
"""

import os
import secrets
from typing import List, Dict, Optional

import pandas as pd

from utils.common_utils import load_config_yaml
from cipher.aes import aesCipher128, aesCipher192, aesCipher256


def _buildCipherByKey(key: bytes):
    """根据密钥长度创建相应的 AES CBC 加密器

    Args:
        key (bytes): 密钥字节串（长度 16/24/32）

    Returns:
        object: 对应的 AES 加密器实例

    Raises:
        ValueError: 当密钥长度不在 16/24/32 字节范围内

    Example:
        >>> c = _buildCipherByKey(secrets.token_bytes(16))
        >>> hasattr(c, 'encrypt') and hasattr(c, 'decrypt')
        True
    """
    # 新增代码：根据密钥长度选择不同 AES 实现
    if len(key) == 16:
        return aesCipher128(key)
    if len(key) == 24:
        return aesCipher192(key)
    if len(key) == 32:
        return aesCipher256(key)
    raise ValueError("仅支持 16/24/32 字节密钥")


def _toHex(data: bytes) -> str:
    """字节串转十六进制字符串（小写不带 0x）

    Args:
        data (bytes): 字节串

    Returns:
        str: 十六进制字符串

    Example:
        >>> _toHex(b"\x00\x01")
        '0001'
    """
    # 新增代码：统一转换为 hex 便于数据落盘与分析
    return data.hex()


def _randomPlaintext(block_size: int) -> bytes:
    """生成一个随机明文样本（长度等于分组大小）

    Args:
        block_size (int): 分组大小（字节），AES 为 16

    Returns:
        bytes: 随机明文字节串

    Example:
        >>> len(_randomPlaintext(16))
        16
    """
    # 新增代码：使用 cryptographically secure 随机源生成明文
    return secrets.token_bytes(block_size)


def generateAesCipherDataset(
    key_size_bytes: int,
    batch_num: int,
    sample_per_batch: int,
    block_size: int = 16,
    save_dir: Optional[str] = None,
    per_sample_random_key: bool = True,
) -> pd.DataFrame:
    """生成单一密钥长度的 AES CBC 密文数据集

    Args:
        key_size_bytes (int): 密钥长度（字节），可为 16/24/32
        batch_num (int): 批次数量
        sample_per_batch (int): 每批次样本数
        block_size (int): 分组大小（字节），默认 16
        save_dir (Optional[str]): 若提供则保存 CSV 至该目录
        per_sample_random_key (bool): 是否每个样本使用新随机密钥

    Returns:
        pd.DataFrame: 包含 `key/iv/pt/ct/key_size/batch_idx/sample_idx` 列的数据集

    Raises:
        ValueError: 参数不合法时抛出

    Example:
        >>> df = generateAesCipherDataset(16, 2, 4)
        >>> set(df.columns) >= {"key", "iv", "pt", "ct"}
        True
    """
    if key_size_bytes not in (16, 24, 32):
        raise ValueError("key_size_bytes 仅支持 16/24/32")

    rows: List[Dict[str, str]] = []
    # 新增代码：为每批次/样本生成密文，按需更换密钥
    for b_idx in range(batch_num):
        base_key = secrets.token_bytes(key_size_bytes)
        for s_idx in range(sample_per_batch):
            key = secrets.token_bytes(key_size_bytes) if per_sample_random_key else base_key
            cipher = _buildCipherByKey(key)
            pt = _randomPlaintext(block_size)
            ct, iv = cipher.encrypt(pt)
            rows.append(
                {
                    "key": _toHex(key),
                    "iv": _toHex(iv),
                    "pt": _toHex(pt),
                    "ct": _toHex(ct),
                    "key_size": str(key_size_bytes * 8),
                    "batch_idx": str(b_idx),
                    "sample_idx": str(s_idx),
                }
            )

    df = pd.DataFrame(rows)

    # 新增代码：按需保存为 CSV 文件
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"aes_{key_size_bytes * 8}_ciphertexts.csv")
        df.to_csv(out_path, index=False)

    return df


def generateAesCipherDatasetsFromConfig(
    config_path: str = "configs/common.yaml",
    per_sample_random_key: bool = True,
) -> Dict[int, pd.DataFrame]:
    """根据 YAML 配置批量生成 AES CBC 密文数据集并保存

    Args:
        config_path (str): 配置文件路径
        per_sample_random_key (bool): 是否每个样本使用新随机密钥

    Returns:
        Dict[int, pd.DataFrame]: key 为密钥位数，value 为对应数据集

    Raises:
        FileNotFoundError: 当配置路径不存在时

    Example:
        >>> datasets = generateAesCipherDatasetsFromConfig()
        >>> 128 in datasets and 256 in datasets
        True
    """
    cfg = load_config_yaml(config_path)
    block_size = int(cfg.get("BLOCK_SIZE", 16))
    sample_per_batch = int(cfg.get("SAMPLE_PER_BATCH", 4096))
    batch_num = int(cfg.get("BATCH_NUM", 1))
    key_sizes = list(cfg.get("AES_KEY_SIZES", [16]))

    save_dir = None
    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        save_dir = data_cfg.get("save_dir_aes")

    results: Dict[int, pd.DataFrame] = {}
    # 新增代码：遍历所有密钥长度生成对应数据集
    for key_sz in key_sizes:
        df = generateAesCipherDataset(
            key_size_bytes=int(key_sz),
            batch_num=batch_num,
            sample_per_batch=sample_per_batch,
            block_size=block_size,
            save_dir=save_dir,
            per_sample_random_key=per_sample_random_key,
        )
        results[key_sz * 8] = df

    return results
