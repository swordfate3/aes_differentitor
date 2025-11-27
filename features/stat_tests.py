import numpy as np

# import pandas as pd

# from typing import Any
from scipy import stats  # type: ignore[import-not-found]
from scipy.stats import (  # type: ignore[import-not-found]
    anderson,
    jarque_bera,
    ks_2samp,
    shapiro,
    mannwhitneyu,
    chi2,
    norm,
)

try:
    from pyentrp import entropy as ent  # type: ignore[import-not-found]
except Exception:
    ent = None  # 新增代码【FIX-ADDED】：pyentrp 不可用时降级为本地实现
from utils.common_utils import load_config_yaml

config = load_config_yaml()
# 新增代码：预计算位长度，避免各测试中重复写死 128
BIT_LEN = int(config.get("BLOCK_SIZE", 16)) * 8


# 新增代码：统一整数样本转定长二进制字符串
def _binaryStr(x: int) -> str:
    """整数转换为固定长度二进制字符串

    Args:
        x (int): 需要转换的整数样本

    Returns:
        str: 定长二进制字符串，长度为配置中的位长度（BLOCK_SIZE*8）

    Example:
        >>> _binaryStr(5).endswith("101")
        True
    """
    return np.binary_repr(x, width=BIT_LEN)


def test_anderson_darling(sample_batch):
    """1. 安德森-达林测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: AD 统计量

    Example:
        >>> test_anderson_darling([1,2,3]) >= 0
        True
    """
    sample = np.array(sample_batch, dtype=np.float64)
    return anderson(sample, dist="norm").statistic


def test_approx_entropy(sample_batch, m=2, r_ratio=0.2):
    """2. 近似熵测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        m (int): 模式长度
        r_ratio (float): 容差与样本标准差的比例

    Returns:
        float: 近似熵值
    """
    binary_seq = "".join([_binaryStr(x) for x in sample_batch])
    binary_arr = np.array([int(bit) for bit in binary_seq], dtype=np.uint8)
    std = float(np.std(binary_arr))
    r = r_ratio * std if std != 0 else 0.1
    if ent is not None and hasattr(ent, "approximate_entropy"):
        # 新增代码【FIX-ADDED】：优先使用 pyentrp 的实现
        return float(ent.approximate_entropy(binary_arr, m, r))

    # 新增代码【FIX-ADDED】：高效本地近似熵实现（二值序列，r<1 等价为模式完全匹配）
    def _phi_exact(seq: np.ndarray, mm: int) -> float:
        n = len(seq)
        if n <= mm:
            return 0.0
        try:
            from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
        except Exception:
            # 退化实现（稍慢，但避免依赖问题）
            windows = np.array(
                [seq[i:i + mm] for i in range(n - mm + 1)], dtype=np.uint8
            )
        else:
            windows = sliding_window_view(seq, mm)
        # 将窗口视为 mm 位二进制数，计算频次
        weights = 2 ** np.arange(mm - 1, -1, -1, dtype=np.uint8)
        vals = (windows * weights).sum(axis=1)
        N = n - mm + 1
        counts = np.bincount(vals, minlength=2**mm)
        # phi_m = (1/N) * sum_i c_i * log(c_i / N)
        p = counts.astype(np.float64) / float(N)
        nonzero = p > 0
        return float(np.sum(counts[nonzero] * np.log(p[nonzero])) / float(N))

    # 当 r < 1（在二值序列上常见）使用快速精确法，否则使用简化近似法
    if r < 1.0:
        phi_m = _phi_exact(binary_arr, m)
        phi_m1 = _phi_exact(binary_arr, m + 1)
        return float(phi_m - phi_m1)
    else:
        # 简化近似法：使用 mm 位窗口的 L2 距离近邻统计（采样降维）
        # 由于 r >= 1 在二值序列较少见，这里采用轻量近似避免阻塞
        max_len = min(len(binary_arr), 65536)
        seq = binary_arr[:max_len]

        def _phi_approx(seq: np.ndarray, mm: int, rr: float) -> float:
            n = len(seq)
            if n <= mm:
                return 0.0
            step = max(1, n // 4096)
            windows = np.array(
                [seq[i:i + mm] for i in range(0, n - mm + 1, step)], dtype=np.float64
            )
            N = windows.shape[0]
            # 计算成对 L2 距离并统计近邻比
            dists = np.sqrt(
                ((windows[:, None, :] - windows[None, :, :]) ** 2).sum(axis=2)
            )
            c = (dists <= rr).sum(axis=1) / float(N)
            c[c == 0] = 1e-12
            return float(np.mean(np.log(c)))

        phi_m = _phi_approx(seq, m, r)
        phi_m1 = _phi_approx(seq, m + 1, r)
        return float(phi_m - phi_m1)


def test_chi2_dist(sample_batch, df=10):
    """3. 卡方分布拟合测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        df (int): 自由度

    Returns:
        float: 卡方统计量
    """
    # 新增代码：使用样本的低64位并进行平方映射，避免溢出与NaN
    mask64 = (1 << 64) - 1
    sample = np.array([int(x) & mask64 for x in sample_batch], dtype=np.float64)
    if sample.size == 0:
        return 0.0
    # 映射到近似卡方分布的非负域：标准化后平方
    std = float(np.std(sample))
    if std == 0.0:
        return 0.0
    sample_z = (sample - float(np.mean(sample))) / std
    sample_chi = sample_z ** 2
    # 直方图与期望频率
    obs, bins = np.histogram(sample_chi, bins=20, density=False)
    centers = (bins[:-1] + bins[1:]) / 2.0
    exp_pdf = chi2.pdf(centers, df=max(1, int(df)))
    exp = exp_pdf / (exp_pdf.sum() + 1e-12) * max(1.0, float(obs.sum()))
    chi2_stat, _ = stats.chisquare(obs, exp)
    return float(chi2_stat)


def test_chi2_unif(sample_batch, bins=20):
    """4. 卡方均匀性测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        bins (int): 直方图分箱数

    Returns:
        float: 卡方统计量
    """
    sample = np.array(sample_batch, dtype=np.float64)
    obs, _ = np.histogram(sample, bins=bins, density=False)
    exp = np.ones_like(obs) * (len(sample) / bins)
    chi2_stat, _ = stats.chisquare(obs, exp)
    return float(chi2_stat)


def test_cumulative_sums(sample_batch):
    """5. 累积和测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 归一化后的最大累积和
    """
    sample = np.array(sample_batch, dtype=np.float64)
    mean = float(np.mean(sample))
    cum_sum = np.cumsum(sample - mean)
    return float(np.max(np.abs(cum_sum)) / np.sqrt(len(sample)))


def test_shannon_entropy(sample_batch):
    """6. 香农熵测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 平均香农熵
    """
    entropies = []
    for x in sample_batch:
        binary = _binaryStr(int(x))
        count0 = binary.count("0")
        count1 = binary.count("1")
        total = BIT_LEN
        if count0 == 0 or count1 == 0:
            entropies.append(0.0)
            continue
        p0 = count0 / total
        p1 = count1 / total
        ent_val = -p0 * np.log2(p0) - p1 * np.log2(p1)
        entropies.append(float(ent_val))
    return float(np.mean(entropies))


def test_f_test(sample_batch, ref_batch):
    """7. F检验（方差比较）

    Args:
        sample_batch (list[int] | np.ndarray): 样本批次
        ref_batch (list[int] | np.ndarray): 参考批次

    Returns:
        float: 方差比（较大方差/较小方差）
    """
    sample = np.array(sample_batch, dtype=np.float64)
    ref = np.array(ref_batch, dtype=np.float64)
    var_sample = float(np.var(sample, ddof=1))
    var_ref = float(np.var(ref, ddof=1))
    return float(max(var_sample, var_ref) / min(var_sample, var_ref))


def test_gap_test(sample_batch, target_bit="1"):
    """8. 间隔测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        target_bit (str): 目标比特字符（"0" 或 "1"）

    Returns:
        float: 间隔标准差与均值比
    """
    binary_seq = "".join([_binaryStr(x) for x in sample_batch])
    target_pos = [i for i, bit in enumerate(binary_seq) if bit == target_bit]
    if len(target_pos) < 2:
        return 0.0
    gaps = np.diff(target_pos)
    mean_gap = float(np.mean(gaps))
    return float(np.std(gaps) / mean_gap) if mean_gap != 0 else 0.0


def test_jarque_bera(sample_batch):
    """9. 雅克-贝拉测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: JB 统计量
    """
    sample = np.array(sample_batch, dtype=np.float64)
    jb_stat, _ = jarque_bera(sample)
    return float(jb_stat)


def test_ks_test(sample_batch, ref_batch):
    """10. 柯尔莫哥洛夫-斯米尔诺夫测试

    Args:
        sample_batch (list[int] | np.ndarray): 样本批次
        ref_batch (list[int] | np.ndarray): 参考批次

    Returns:
        float: KS 统计量
    """
    sample = np.array(sample_batch, dtype=np.float64)
    ref = np.array(ref_batch, dtype=np.float64)
    ks_stat, _ = ks_2samp(sample, ref)
    return float(ks_stat)


def test_lilliefors(sample_batch):
    """11. 利利福斯测试（近似）

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 近似 KS 统计量
    """
    sample = np.array(sample_batch, dtype=np.float64)
    mu = float(np.mean(sample))
    std = float(np.std(sample, ddof=1))
    if std == 0.0:
        return 0.0
    sample_norm = (sample - mu) / std
    sample_norm = sample_norm[np.isfinite(sample_norm)]
    if sample_norm.size < 5:
        return 0.0
    ks_stat, _ = ks_2samp(sample_norm, norm.rvs(size=sample_norm.size, loc=0, scale=1))
    return float(ks_stat)


def test_monobit(sample_batch):
    """12. 单比特测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 单比特偏差统计量
    """
    total_ones = 0
    total_bits = 0
    for x in sample_batch:
        binary = _binaryStr(int(x))
        total_ones += binary.count("1")
        total_bits += BIT_LEN
    zeros = total_bits - total_ones
    return float(np.abs(total_ones - zeros) / np.sqrt(total_bits))


def test_runs_test_blocks(sample_batch, block_size=8):
    """13. 块游程测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        block_size (int): 块大小（字节）

    Returns:
        float: 游程统计量
    """
    block_means = []
    for x in sample_batch:
        bytes_arr = int(x).to_bytes(config["BLOCK_SIZE"], byteorder="big")
        block_means.extend(bytes_arr)
    runs = 1
    for i in range(1, len(block_means)):
        if block_means[i] != block_means[i - 1]:
            runs += 1
    n = len(block_means)
    exp_runs = 2 * n * 0.5 * 0.5 + 1
    return float(np.abs(runs - exp_runs) / np.sqrt(n))


def test_serial_test(sample_batch, m=2):
    """14. 序列测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        m (int): 模式长度

    Returns:
        float: 卡方统计量
    """
    binary_seq = "".join([_binaryStr(x) for x in sample_batch])
    n_patterns = 2**m
    # 新增代码【FIX-CHANGED】：使用滑窗+bincount计算各模式频次，确保期望与观测之和一致
    try:
        from numpy.lib.stride_tricks import sliding_window_view  # type: ignore

        windows = sliding_window_view(
            np.fromiter((int(b) for b in binary_seq), dtype=np.uint8), m
        )
    except Exception:
        seq = np.array([int(b) for b in binary_seq], dtype=np.uint8)
        windows = np.array(
            [seq[i:i + m] for i in range(len(seq) - m + 1)], dtype=np.uint8
        )
    weights = 2 ** np.arange(m - 1, -1, -1, dtype=np.uint8)
    vals = (windows * weights).sum(axis=1)
    obs = np.bincount(vals, minlength=n_patterns).astype(np.float64)
    exp_per_bin = float(len(vals)) / float(n_patterns)
    f_exp = np.full(n_patterns, exp_per_bin, dtype=np.float64)
    # 归一化处理，保证总和完全匹配
    f_exp *= obs.sum() / f_exp.sum()
    chi2_stat, _ = stats.chisquare(obs, f_exp)
    return float(chi2_stat)


def test_shapiro_wilk(sample_batch):
    """15. 夏皮罗-威尔克测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: SW 统计量
    """
    sample = np.array(sample_batch, dtype=np.float64)
    if len(sample) > 5000:
        sample = sample[:5000]
    sw_stat, _ = shapiro(sample)
    return float(sw_stat)


def test_t_test(sample_batch, ref_batch):
    """16. t检验（均值比较）

    Args:
        sample_batch (list[int] | np.ndarray): 样本批次
        ref_batch (list[int] | np.ndarray): 参考批次

    Returns:
        float: t 统计量绝对值
    """
    sample = np.array(sample_batch, dtype=np.float64)
    ref = np.array(ref_batch, dtype=np.float64)
    t_stat, _ = stats.ttest_ind(sample, ref, equal_var=False)
    return float(np.abs(t_stat))


def test_hamming_weight(sample_batch):
    """17. 汉明重量测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 汉明重量的标准差
    """
    weights = []
    for x in sample_batch:
        binary = _binaryStr(int(x))
        weights.append(binary.count("1"))
    return float(np.std(weights))


def test_frequency_block(sample_batch, block_size=16):
    """18. 块频率测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        block_size (int): 频率统计的块大小（比特）

    Returns:
        float: 频率偏差统计量
    """
    block_freqs = []
    blocks_count = max(1, BIT_LEN // block_size)
    for x in sample_batch:
        binary = _binaryStr(int(x))
        blocks = [
            binary[i * block_size:(i + 1) * block_size] for i in range(blocks_count)
        ]
        for block in blocks:
            ones = block.count("1")
            block_freqs.append(ones / block_size)
    return float(4 * block_size * np.sum((np.array(block_freqs) - 0.5) ** 2))


def test_autocorrelation(sample_batch, lag=1):
    """19. 自相关测试

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        lag (int): 滞后步长

    Returns:
        float: 自相关绝对值
    """
    bits = []
    for x in sample_batch:
        binary = _binaryStr(int(x))
        bits.append(int(binary[0]))
    bits = np.array(bits)
    if len(bits) <= lag:
        return 0.0
    corr = float(np.corrcoef(bits[:-lag], bits[lag:])[0, 1])
    return float(np.abs(corr))


def test_mann_whitney(sample_batch, ref_batch):
    """20. 曼-惠特尼 U 测试

    Args:
        sample_batch (list[int] | np.ndarray): 样本批次
        ref_batch (list[int] | np.ndarray): 参考批次

    Returns:
        float: 归一化 U 统计量
    """
    sample = np.atleast_1d(np.asarray(sample_batch, dtype=np.float64)).ravel()
    ref = np.atleast_1d(np.asarray(ref_batch, dtype=np.float64)).ravel()
    u_stat, _ = mannwhitneyu(sample, ref)
    n1, n2 = sample.size, ref.size
    return float(u_stat / (n1 * n2))


# 统计测试函数映射（便于批量调用）
STAT_TEST_FUNCTIONS = [
    test_anderson_darling,
    test_approx_entropy,
    test_chi2_dist,
    test_chi2_unif,
    test_cumulative_sums,
    test_shannon_entropy,
    test_f_test,
    test_gap_test,
    test_jarque_bera,
    test_ks_test,
    test_lilliefors,
    test_monobit,
    test_runs_test_blocks,
    test_serial_test,
    test_shapiro_wilk,
    test_t_test,
    test_hamming_weight,
    test_frequency_block,
    test_autocorrelation,
    test_mann_whitney
]


# 新增代码【FIX-ADDED】：21. 最长连续1游程测试
def test_longest_run_ones(sample_batch):
    """21. 最长连续1游程测试

    计算连接后的二进制序列中，最长的连续“1”游程长度，并归一化。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 最长游程长度的归一化值（除以 `BIT_LEN`）

    Example:
        >>> test_longest_run_ones([0b1110, 0b0111]) > 0
        True
    """
    binary_seq = "".join([_binaryStr(int(x)) for x in sample_batch])
    longest = 0
    current = 0
    for ch in binary_seq:
        if ch == "1":
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return float(longest / float(BIT_LEN))


# 新增代码【FIX-ADDED】：22. 二值矩阵秩测试（32x32 GF(2)）
def test_matrix_rank_32(sample_batch):
    """22. 二值矩阵秩测试（32x32）

    将比特序列切分为多个 32x32 二值矩阵，在 GF(2) 下进行高斯消元，
    统计平均秩并做归一化（除以 32）。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 平均秩的归一化值（0~1）

    Example:
        >>> test_matrix_rank_32([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    nbits = len(binary)
    block = 32 * 32
    if nbits < block:
        return 0.0
    seq = np.fromiter((1 if c == "1" else 0 for c in binary), dtype=np.uint8)
    ranks = []

    def _gf2_rank(mat: np.ndarray) -> int:
        m = mat.copy()
        rows, cols = m.shape
        r = 0
        for c in range(cols):
            pivot = None
            for rr in range(r, rows):
                if m[rr, c] == 1:
                    pivot = rr
                    break
            if pivot is None:
                continue
            if pivot != r:
                m[[r, pivot]] = m[[pivot, r]]
            for rr in range(rows):
                if rr != r and m[rr, c] == 1:
                    m[rr, :] ^= m[r, :]
            r += 1
            if r == rows:
                break
        return r

    total_blocks = nbits // block
    for i in range(total_blocks):
        chunk = seq[i * block:(i + 1) * block]
        mat = chunk.reshape(32, 32)
        ranks.append(_gf2_rank(mat))
    if not ranks:
        return 0.0
    return float(np.mean(ranks) / 32.0)


# 新增代码【FIX-ADDED】：23. 频谱DFT测试（简化版）
def test_dft_spectral(sample_batch):
    """23. 频谱DFT测试（简化）

    将比特映射到 {+1, -1}，计算离散傅里叶幅值。
    统计幅值低于阈值 `T = sqrt(log(1/0.05) * n)` 的比例。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 低幅值峰比例（0~1）

    Example:
        >>> test_dft_spectral([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    if not binary:
        return 0.0
    seq = np.fromiter((1 if c == "1" else -1 for c in binary), dtype=np.int8)
    n = seq.size
    spec = np.fft.rfft(seq)
    mags = np.abs(spec)
    T = float(np.sqrt(np.log(1.0 / 0.05) * n))
    prop = float((mags < T).sum()) / float(mags.size)
    return prop


# 新增代码【FIX-ADDED】：24. 非重叠模板匹配（模板："111"）
def test_nonoverlap_template(sample_batch, template="111"):
    """24. 非重叠模板匹配测试

    统计非重叠窗口中模板出现次数，相对期望进行归一化。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        template (str): 模板串（例如 "111"）

    Returns:
        float: 出现次数与期望之比

    Example:
        >>> test_nonoverlap_template([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    m = len(template)
    if m == 0:
        return 0.0
    count = 0
    i = 0
    while i + m <= len(binary):
        if binary[i:i + m] == template:
            count += 1
            i += m
        else:
            i += 1
    expected = max(1.0, (len(binary) - m + 1) / m / (2 ** m))
    return float(count) / float(expected)


# 新增代码【FIX-ADDED】：25. 重叠模板匹配（模板："101"）
def test_overlap_template(sample_batch, template="101"):
    """25. 重叠模板匹配测试

    统计重叠窗口中模板出现次数，相对期望进行归一化。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）
        template (str): 模板串（例如 "101"）

    Returns:
        float: 出现次数与期望之比

    Example:
        >>> test_overlap_template([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    m = len(template)
    if m == 0 or len(binary) < m:
        return 0.0
    count = 0
    for i in range(len(binary) - m + 1):
        if binary[i:i + m] == template:
            count += 1
    expected = max(1.0, (len(binary) - m + 1) / (2 ** m))
    return float(count) / float(expected)


# 新增代码【FIX-ADDED】：26. 线性复杂度测试（Berlekamp–Massey）
def test_linear_complexity(sample_batch):
    """26. 线性复杂度测试（BM算法）

    对每个块的比特序列运行 Berlekamp–Massey 算法，
    计算线性复杂度并返回其相对位长的平均值。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 平均线性复杂度的归一化值（除以 `BIT_LEN`）

    Example:
        >>> test_linear_complexity([1, 2, 3]) >= 0
        True
    """
    def bm_complexity(bits: np.ndarray) -> int:
        n = bits.size
        C = np.zeros(n + 1, dtype=np.uint8)
        B = np.zeros(n + 1, dtype=np.uint8)
        C[0] = 1
        B[0] = 1
        L = 0
        m = -1
        for N in range(n):
            d = bits[N]
            for i in range(1, L + 1):
                d ^= (C[i] & bits[N - i])
            if d == 1:
                T = C.copy()
                shift = N - m
                for i in range(n - shift + 1):
                    C[i + shift] ^= B[i]
                if 2 * L <= N:
                    L = N + 1 - L
                    m = N
                    B = T
        return L

    comps = []
    for x in sample_batch:
        bstr = _binaryStr(int(x))
        bits = np.fromiter((1 if c == "1" else 0 for c in bstr), dtype=np.uint8)
        comps.append(bm_complexity(bits))
    if not comps:
        return 0.0
    return float(np.mean(comps) / float(BIT_LEN))


# 新增代码【FIX-ADDED】：27. Maurer通用性测试（简化LZ压缩近似）
def test_maurer_universal(sample_batch):
    """27. Maurer通用性测试（简化）

    使用 LZ78 风格的简化字典压缩估计可压缩性，返回压缩率。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 平均压缩率（压缩后长度/原始长度）

    Example:
        >>> test_maurer_universal([1, 2, 3]) >= 0
        True
    """
    def lz78_ratio(s: str) -> float:
        dict_idx = {"": 0}
        idx = 1
        w = ""
        output = []
        for c in s:
            wc = w + c
            if wc in dict_idx:
                w = wc
            else:
                output.append((dict_idx[w], c))
                dict_idx[wc] = idx
                idx += 1
                w = ""
        if w:
            output.append((dict_idx[w], ""))
        return float(len(output)) / float(max(1, len(s)))

    ratios = []
    for x in sample_batch:
        bstr = _binaryStr(int(x))
        ratios.append(lz78_ratio(bstr))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


# 新增代码【FIX-ADDED】：28. 随机游走游程测试（Random Excursions）
def test_random_excursions(sample_batch):
    """28. 随机游走游程测试（简化）

    将比特映射为 {+1,-1} 并累积为随机游走，统计状态
    -4..+4（不含0）访问次数的方差，返回归一化方差。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 状态访问次数的归一化方差

    Example:
        >>> test_random_excursions([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    if not binary:
        return 0.0
    seq = np.fromiter((1 if c == "1" else -1 for c in binary), dtype=np.int16)
    walk = np.cumsum(seq)
    visits = []
    for s in range(-4, 5):
        if s == 0:
            continue
        visits.append(int((walk == s).sum()))
    v = np.array(visits, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return float(np.var(v) / float(len(binary)))


# 新增代码【FIX-ADDED】：29. 随机游走变体测试（Random Excursions Variant）
def test_random_excursions_variant(sample_batch):
    """29. 随机游走变体测试（简化）

    基于随机游走，统计各状态的平均停留次数并返回其
    标准差的归一化值。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: 状态平均停留次数的标准差/总步数

    Example:
        >>> test_random_excursions_variant([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    if not binary:
        return 0.0
    seq = np.fromiter((1 if c == "1" else -1 for c in binary), dtype=np.int16)
    walk = np.cumsum(seq)
    means = []
    for s in range(-4, 5):
        if s == 0:
            continue
        idx = np.where(walk == s)[0]
        if idx.size == 0:
            means.append(0.0)
        else:
            gaps = np.diff(idx)
            means.append(float(np.mean(gaps)) if gaps.size > 0 else 0.0)
    m = np.array(means, dtype=np.float64)
    return float(np.std(m) / float(len(binary)))


# 新增代码【FIX-ADDED】：30. 比特跃迁率测试
def test_bit_transitions(sample_batch):
    """30. 比特跃迁率测试

    统计相邻比特位发生变化的比例，返回与 0.5 的偏差。

    Args:
        sample_batch (list[int] | np.ndarray): 批次样本（整数）

    Returns:
        float: |跃迁比例 - 0.5|

    Example:
        >>> test_bit_transitions([1, 2, 3]) >= 0
        True
    """
    binary = "".join([_binaryStr(int(x)) for x in sample_batch])
    if len(binary) < 2:
        return 0.0
    arr = np.fromiter((1 if c == "1" else 0 for c in binary), dtype=np.uint8)
    trans = np.sum(arr[1:] ^ arr[:-1])
    prop = float(trans) / float(len(arr) - 1)
    return float(abs(prop - 0.5))


# 新增代码【FIX-ADDED】：将新增测试加入映射列表
STAT_TEST_FUNCTIONS.extend([
    test_longest_run_ones,
    test_matrix_rank_32,
    test_dft_spectral,
    test_nonoverlap_template,
    test_overlap_template,
    test_linear_complexity,
    test_maurer_universal,
    test_random_excursions,
    test_random_excursions_variant,
    test_bit_transitions,
])
