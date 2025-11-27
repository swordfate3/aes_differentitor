import numpy as np
from .AES128 import AES128


class AES256(AES128):
    """AES-256 加密实现（继承 AES128，重写密钥扩展）

    采用位数组接口：明文 128 位、密钥 256 位，返回密文 128 位。
    仅实现单块加密流程（无模式/填充）。

    Args:
        rounds (int): 轮数，默认 14（有效范围 [1, 14]）

    Raises:
        ValueError: 当轮数不在有效范围时抛出

    Example:
        >>> import numpy as np
        >>> aes = AES256()
        >>> pt = np.zeros(128, dtype=np.uint8)
        >>> k = np.zeros(256, dtype=np.uint8)
        >>> ct = aes.encrypt(pt, k)
        >>> ct.size == 128
        True
    """

    def __init__(self, rounds: int = 14):
        super().__init__(rounds=rounds)

    def _key_schedule(self, key_bits: np.ndarray) -> list:
        """密钥扩展（AES-256）生成所有轮密钥

        Args:
            key_bits (np.ndarray): 256 位密钥位数组（uint8 0/1）

        Returns:
            list[np.ndarray]: 轮密钥列表，每项为 16 字节数组

        Raises:
            ValueError: 当密钥长度不是 256 位时抛出
        """
        if key_bits.size != 256:
            raise ValueError("AES256 仅支持 256 位密钥")
        key_bytes = np.packbits(key_bits, bitorder='big')
        Nk = 8
        Nr = max(1, min(self.rounds, 14))
        Nb = 4
        total_words = Nb * (Nr + 1)

        w = [
            np.array(key_bytes[4*i:4*(i+1)], dtype=np.uint8)
            for i in range(Nk)
        ]
        i = Nk
        while len(w) < total_words:
            temp = w[-1].copy()
            if i % Nk == 0:
                temp = self._sub_word(self._rot_word(temp)).copy()
                temp[0] ^= int(self.RCON[i // Nk] & 0xFF)
            elif i % Nk == 4:
                temp = self._sub_word(temp).copy()
            w.append(np.bitwise_xor(w[i - Nk], temp))
            i += 1

        round_keys = []
        for r in range(Nr + 1):
            rk = np.concatenate(w[r*Nb:(r+1)*Nb])
            round_keys.append(rk.astype(np.uint8))
        return round_keys