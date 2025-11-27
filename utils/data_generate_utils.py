import os
import secrets
import numpy as np
from utils.common_utils import load_config_yaml
from cipher import AES128  # 新增代码：使用仓库内自定义AES实现（可控轮数）

config = load_config_yaml()


def int_to_128bit_binary(x):
    """整数转128位二进制字符串"""
    return np.binary_repr(x, width=128)


def generate_random_data():
    """生成4个独立随机数据集（每个2^21样本）"""

    random_datasets = []
    for _ in range(config["RANDOM_SET_NUM"]):
        dataset = []
        for _ in range(config["BATCH_NUM"]):
            # 每个样本为128bit安全随机整数
            batch = [secrets.randbits(128) for _ in range(config["SAMPLE_PER_BATCH"])]
            dataset.append(batch)
        random_datasets.append(dataset)
    return random_datasets


# def generate_aes_ciphertexts(key_size: int = 32, is_Multi: bool = True):
#     """生成AES密文（PyCryptodome，ECB，全轮）

#     使用 PyCryptodome 的 AES 块加密（ECB），支持 128/192/256 位密钥。
#     论文要求“全轮”，PyCryptodome默认为全轮实现，满足要求。

#     Args:
#         key_size (int): 密钥长度（字节），支持 16/24/32，默认 32（AES-256）
#         is_Multi (bool): 多密钥开关，True 为多密钥，False 为单密钥

#     Returns:
#         list[list[int]]: 每批次的密文整数列表（按批次返回）
#     """
#     print(f"生成 AES-{key_size*8} 密文（{'多密钥' if is_Multi else '单密钥'}，ECB）")
#     if key_size not in (16, 24, 32):
#         raise ValueError("key_size 仅支持 16/24/32 字节 (AES-128/192/256)")
#     results = []

#     # 单密钥（若需要）
#     single_key = secrets.token_bytes(key_size)
#     for _ in range(config["BATCH_NUM"]):
#         batch_out = []
#         for _ in range(config["SAMPLE_PER_BATCH"]):
#             pt = secrets.token_bytes(config["BLOCK_SIZE"])  # 16B
#             if is_Multi:
#                 key = secrets.token_bytes(key_size)
#             else:
#                 key = single_key
#             cipher = AES.new(key, AES.MODE_ECB)
#             ct = cipher.encrypt(pt)
#             batch_out.append(int.from_bytes(ct, byteorder="big"))
#         results.append(batch_out)
#     print(
#         f"共生成 {config['BATCH_NUM']} 批次，每批次 {config['SAMPLE_PER_BATCH']} 样本"
#     )
#     return results


def generateAesCiphertextsBatch(key_size: int = 16, is_Multi: bool = True, rounds: int = 10):
    """生成一批次 AES 密文（位数组实现，支持单密钥/多密钥）

    使用位数组接口的 AES128/192/256，对当前批次的 `SAMPLE_PER_BATCH` 个样本加密，
    返回该批次的密文整数列表。适合在保存流程中追加式写入，避免一次性占用内存。

    Args:
        key_size (int): 密钥长度（字节），支持 16/24/32
        is_Multi (bool): 多密钥开关，True 为多密钥，False 为单密钥

    Returns:
        list[int]: 一个批次的密文整数列表

    Raises:
        ValueError: key_size 非法时抛出
    """
    if key_size != 16:
        raise ValueError("当前自定义AES实现仅支持128位块/128位密钥")

    # 生成明文并转为位数组
    pts = [
        secrets.token_bytes(config["BLOCK_SIZE"])
        for _ in range(config["SAMPLE_PER_BATCH"])
    ]
    pt_bits_list = [
        np.unpackbits(np.frombuffer(pt, dtype=np.uint8), bitorder="big") for pt in pts
    ]

    # 单密钥位数组（若关闭多密钥）
    single_key_bits = None
    if not is_Multi:
        sk = secrets.token_bytes(key_size)
        single_key_bits = np.unpackbits(
            np.frombuffer(sk, dtype=np.uint8), bitorder="big"
        )

    aes = AES128(rounds=rounds)
    out = []
    for pt_bits in pt_bits_list:
        if is_Multi:
            mk = secrets.token_bytes(key_size)
            mk_bits = np.unpackbits(np.frombuffer(mk, dtype=np.uint8), bitorder="big")
            ct_bits = aes.encrypt(pt_bits, mk_bits)
        else:
            ct_bits = aes.encrypt(pt_bits, single_key_bits)  # type: ignore
        ct_bytes = np.packbits(ct_bits, bitorder="big")
        out.append(int.from_bytes(ct_bytes.tobytes(), byteorder="big"))
    # print(f"{np.array(out).shape}")
    return out


def generate_and_save_data():
    """追加式生成并保存数据，避免一次性占用内存

    - 随机数据、AES-128 单密钥、多密钥均按批次生成并单独保存为 .npy 文件
    - 最后写入索引 .npz 文件，记录路径与元信息，便于后续加载
    """
    save_dir = config["data"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_index = os.path.join(save_dir, "aes_datasets.npz")

    rand_dir = os.path.join(save_dir, "random_batches")
    single_dir = os.path.join(save_dir, "aes128_custom_single")
    multi_dir = os.path.join(save_dir, "aes128_custom_multi")
    os.makedirs(rand_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    batch_num = int(config["BATCH_NUM"])
    sample_per_batch = int(config["SAMPLE_PER_BATCH"])
    block_size = int(config["BLOCK_SIZE"])

    print("生成随机数据集...")
    print("生成自定义AES密文（可控轮数，单密钥）...")
    print("生成自定义AES密文（可控轮数，多密钥）...")

    for i in range(batch_num):
        # 随机批次（每样本 block_size*8 位整数）
        rand_batch = [secrets.randbits(block_size * 8) for _ in range(sample_per_batch)]
        np.save(
            os.path.join(rand_dir, f"batch_{i}.npy"),
            np.asarray(rand_batch, dtype=object),
        )

        # 自定义AES-128（可控轮数）单密钥与多密钥批次
        rounds = int(config.get("ROUNDS", 10))
        batch_single = generateAesCiphertextsBatch(16, is_Multi=False, rounds=rounds)
        batch_multi = generateAesCiphertextsBatch(16, is_Multi=True, rounds=rounds)
        np.save(
            os.path.join(single_dir, f"batch_{i}.npy"),
            np.asarray(batch_single, dtype=object),
        )
        np.save(
            os.path.join(multi_dir, f"batch_{i}.npy"),
            np.asarray(batch_multi, dtype=object),
        )

        print(f"已写入批次 {i+1}/{batch_num}")

    # 合并批次文件到单个 npz 并删除临时 .npy
    def _load_dir(d):
        files = sorted([f for f in os.listdir(d) if f.endswith(".npy")])
        arrs = [np.load(os.path.join(d, f), allow_pickle=True) for f in files]
        return np.array(arrs, dtype=object)

    random_all = _load_dir(rand_dir)
    single_all = _load_dir(single_dir)
    multi_all = _load_dir(multi_dir)

    np.savez(
        save_index,
        random_datasets=random_all,
        aes128_custom_single=single_all,
        aes128_custom_multi=multi_all,
        meta={
            "batch_num": batch_num,
            "sample_per_batch": sample_per_batch,
            "block_size": block_size,
            "rounds": int(config.get("AES_ROUNDS", 10)),
        },
    )

    # 删除临时目录
    for d in (rand_dir, single_dir, multi_dir):
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            try:
                os.remove(fp)
            except Exception:
                pass
        try:
            os.rmdir(d)
        except Exception:
            pass

    print(f"数据保存完成并合并：{save_index}")


def load_data():
    """加载合并后的数据集（单个 npz 文件）"""
    index_path = os.path.join(config["data"]["save_dir"], "aes_datasets.npz")
    if not os.path.exists(index_path):
        raise FileNotFoundError("数据集未生成，请先运行generate_and_save_data()")
    data = np.load(index_path, allow_pickle=True)
    keys = set(data.files)
    out = {
        "random_datasets": data["random_datasets"],
        "meta": data["meta"].item(),
    }
    # 兼容键名
    for k in ("aes256_single", "aes256_multi", "aes128_single", "aes128_multi",
              "aes128_custom_single", "aes128_custom_multi"):
        if k in keys:
            out[k] = data[k]
    return out


if __name__ == "__main__":
    generate_and_save_data()
    load_data()
