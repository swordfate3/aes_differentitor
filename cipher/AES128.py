import numpy as np


class AES128:
    """完整 AES-128 加密实现（位数组接口，轮数可配置）

    - 明文：长度 128 的位数组（np.uint8，0/1）
    - 密钥：长度 128 的位数组（np.uint8，0/1）
    - 轮数：默认10，最多支持14
    """
    def __init__(self, rounds: int = 10):
        """初始化 AES-128 算法对象

        Args:
            rounds (int): 轮数，默认 10（最多支持 14）

        Raises:
            ValueError: 当轮数不在有效范围时抛出

        Example:
            >>> aes = AES128(rounds=10)
        """
        self.rounds = int(rounds)
        print(f"初始化 AES-128 算法对象，轮数={self.rounds}")
        if not (1 <= self.rounds <= 14):
            raise ValueError("轮数必须在 [1, 14] 范围内")
        self.S_BOX = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
            0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
            0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
            0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
            0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,  # type: ignore
            0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
            0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
            0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
            0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
            0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
            0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
            0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
            0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
            0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
            0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
            0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
            0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
        ], dtype=np.uint8)
        self.RCON = np.array(
            [
                0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
                0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D,
            ],
            dtype=np.uint8,
        )
        # 新增代码：构造逆 S-盒，用于解密过程
        inv = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv[int(self.S_BOX[i])] = i
        self.INV_S_BOX = inv

    # ===== 位/字节转换 =====
    @staticmethod
    def _bytes_from_bits(bits: np.ndarray) -> np.ndarray:
        """位数组转字节数组（128 位 -> 16 字节）

        Args:
            bits (np.ndarray): 一维位数组，长度 128，元素为 0/1

        Returns:
            np.ndarray: 长度 16 的字节数组（uint8）

        Raises:
            ValueError: 当位数组形状或长度不合法时抛出

        Example:
            >>> import numpy as np
            >>> AES128._bytes_from_bits(np.zeros(128, dtype=np.uint8)).size == 16
            True
        """
        if bits.ndim != 1 or bits.size != 128:
            raise ValueError("AES128 需要长度为128的一维位数组")
        if bits.dtype != np.uint8:
            bits = bits.astype(np.uint8)
        # 新增代码：使用 packbits 向量化转换
        return np.packbits(bits.reshape(16, 8), axis=1, bitorder='big').reshape(16)

    @staticmethod
    def _bits_from_bytes(bytes_arr: np.ndarray) -> np.ndarray:
        """字节数组转位数组（16 字节 -> 128 位）

        Args:
            bytes_arr (np.ndarray): 一维字节数组，长度 16（uint8）

        Returns:
            np.ndarray: 长度 128 的位数组（uint8，元素 0/1）

        Example:
            >>> import numpy as np
            >>> AES128._bits_from_bytes(np.zeros(16, dtype=np.uint8)).size == 128
            True
        """
        if bytes_arr.ndim != 1 or bytes_arr.size != 16:
            raise ValueError("需要长度为16的一维字节数组")
        # 新增代码：使用 unpackbits 向量化转换
        return np.unpackbits(bytes_arr, bitorder='big')

    # ===== 组件 =====
    def _sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """字节替换（S-盒）

        Args:
            state (np.ndarray): 16 字节的状态数组

        Returns:
            np.ndarray: 16 字节的替换后状态
        """
        return self.S_BOX[state]

    # 新增代码：逆字节替换（逆 S-盒）
    def _inv_sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """逆字节替换（逆 S-盒）

        Args:
            state (np.ndarray): 16 字节的状态数组

        Returns:
            np.ndarray: 16 字节的替换后状态
        """
        return self.INV_S_BOX[state]

    @staticmethod
    def _shift_rows(state: np.ndarray) -> np.ndarray:
        """行移位（按 AES 规则循环左移第 r 行 r 位）

        Args:
            state (np.ndarray): 16 字节状态

        Returns:
            np.ndarray: 行移位后的状态
        """
        # 新增代码：向量化矩阵重排与行移位
        mat = state.reshape(4, 4, order='F').copy()
        for r in range(4):
            mat[r] = np.roll(mat[r], -r)
        return mat.reshape(16, order='F')

    # 新增代码：逆行移位（按 AES 规则循环右移第 r 行 r 位）
    @staticmethod
    def _inv_shift_rows(state: np.ndarray) -> np.ndarray:
        """逆行移位（按 AES 规则循环右移第 r 行 r 位）

        Args:
            state (np.ndarray): 16 字节状态

        Returns:
            np.ndarray: 逆行移位后的状态
        """
        mat = state.reshape(4, 4, order='F').copy()
        for r in range(4):
            mat[r] = np.roll(mat[r], r)
        return mat.reshape(16, order='F')

    @staticmethod
    def _xtime(b: int) -> int:
        b <<= 1
        if b & 0x100:
            b ^= 0x11B
        return b & 0xFF

    @classmethod
    def _mul2(cls, b: int) -> int:
        return cls._xtime(b)

    @classmethod
    def _mul3(cls, b: int) -> int:
        return cls._mul2(b) ^ (b & 0xFF)

    @classmethod
    def _mix_columns(cls, state: np.ndarray) -> np.ndarray:
        """列混淆（GF(2^8) 线性变换）

        Args:
            state (np.ndarray): 16 字节状态

        Returns:
            np.ndarray: 变换后的状态
        """
        out = np.zeros(16, dtype=np.uint8)
        for c in range(4):
            s0 = int(state[4*c + 0])
            s1 = int(state[4*c + 1])
            s2 = int(state[4*c + 2])
            s3 = int(state[4*c + 3])
            out[4*c + 0] = cls._mul2(s0) ^ cls._mul3(s1) ^ s2 ^ s3
            out[4*c + 1] = s0 ^ cls._mul2(s1) ^ cls._mul3(s2) ^ s3
            out[4*c + 2] = s0 ^ s1 ^ cls._mul2(s2) ^ cls._mul3(s3)
            out[4*c + 3] = cls._mul3(s0) ^ s1 ^ s2 ^ cls._mul2(s3)
        return out

    # 新增代码：GF(2^8) 乘法辅助（4、8）与逆列混淆所需常数乘法
    @classmethod
    def _mul4(cls, b: int) -> int:
        return cls._xtime(cls._xtime(b))

    @classmethod
    def _mul8(cls, b: int) -> int:
        return cls._xtime(cls._mul4(b))

    @classmethod
    def _mul9(cls, b: int) -> int:
        return cls._mul8(b) ^ (b & 0xFF)

    @classmethod
    def _mul11(cls, b: int) -> int:
        return cls._mul8(b) ^ cls._mul2(b) ^ (b & 0xFF)

    @classmethod
    def _mul13(cls, b: int) -> int:
        return cls._mul8(b) ^ cls._mul4(b) ^ (b & 0xFF)

    @classmethod
    def _mul14(cls, b: int) -> int:
        return cls._mul8(b) ^ cls._mul4(b) ^ cls._mul2(b)

    # 新增代码：逆列混淆
    @classmethod
    def _inv_mix_columns(cls, state: np.ndarray) -> np.ndarray:
        """逆列混淆（GF(2^8) 线性逆变换）

        Args:
            state (np.ndarray): 16 字节状态

        Returns:
            np.ndarray: 逆变换后的状态
        """
        out = np.zeros(16, dtype=np.uint8)
        for c in range(4):
            s0 = int(state[4*c + 0])
            s1 = int(state[4*c + 1])
            s2 = int(state[4*c + 2])
            s3 = int(state[4*c + 3])
            out[4*c + 0] = (
                cls._mul14(s0) ^ cls._mul11(s1) ^ cls._mul13(s2) ^ cls._mul9(s3)
            )
            out[4*c + 1] = (
                cls._mul9(s0) ^ cls._mul14(s1) ^ cls._mul11(s2) ^ cls._mul13(s3)
            )
            out[4*c + 2] = (
                cls._mul13(s0) ^ cls._mul9(s1) ^ cls._mul14(s2) ^ cls._mul11(s3)
            )
            out[4*c + 3] = (
                cls._mul11(s0) ^ cls._mul13(s1) ^ cls._mul9(s2) ^ cls._mul14(s3)
            )
        return out

    @staticmethod
    def _add_round_key(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(state, round_key)

    # ===== 密钥扩展 =====
    def _rot_word(self, word: np.ndarray) -> np.ndarray:
        return np.array([word[1], word[2], word[3], word[0]], dtype=np.uint8)

    def _sub_word(self, word: np.ndarray) -> np.ndarray:
        return self.S_BOX[word]

    def _key_schedule(self, key_bits: np.ndarray) -> list:
        """密钥扩展，生成所有轮密钥

        Args:
            key_bits (np.ndarray): 128 位密钥位数组（uint8 0/1）

        Returns:
            list[np.ndarray]: 轮密钥列表，每项为 16 字节数组
        """
        key_bytes = self._bytes_from_bits(key_bits)
        Nk = 4
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
            w.append(np.bitwise_xor(w[i - Nk], temp))
            i += 1

        round_keys = []
        for r in range(Nr + 1):
            rk = np.concatenate(w[r*Nb:(r+1)*Nb])
            round_keys.append(rk.astype(np.uint8))
        return round_keys

    # ===== 加密 =====
    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """AES 单块加密（位数组接口）

        Args:
            plaintext (np.ndarray): 128 位明文位数组（uint8 0/1）
            key (np.ndarray): 密钥位数组（uint8 0/1），支持 128/192/256 位

        Returns:
            np.ndarray: 128 位密文字（uint8 0/1）

        Raises:
            ValueError: 当明文长度不为 128 位或密钥长度非法时抛出

        Example:
            >>> import numpy as np
            >>> aes = AES128()
            >>> pt = np.zeros(128, dtype=np.uint8)
            >>> k = np.zeros(128, dtype=np.uint8)
            >>> ct = aes.encrypt(pt, k)
            >>> ct.size == 128
            True
        """
        # 新增代码：允许 128/192/256 位密钥，明文固定 128 位
        if plaintext.size != 128:
            raise ValueError("明文必须为 128 位")
        if key.size not in (128, 192, 256):
            raise ValueError("密钥长度必须为 128/192/256 位")
        state = self._bytes_from_bits(plaintext)
        round_keys = self._key_schedule(key)
        Nr = len(round_keys) - 1

        state = self._add_round_key(state, round_keys[0])

        for r in range(1, Nr):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, round_keys[r])

        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, round_keys[Nr])

        return self._bits_from_bytes(state)

    # 新增代码：解密
    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """AES 单块解密（位数组接口）

        Args:
            ciphertext (np.ndarray): 128 位密文字（uint8 0/1）
            key (np.ndarray): 密钥位数组（uint8 0/1），支持 128/192/256 位

        Returns:
            np.ndarray: 128 位明文字（uint8 0/1）

        Raises:
            ValueError: 当密文长度不为 128 位或密钥长度非法时抛出

        Example:
            >>> import numpy as np
            >>> aes = AES128()
            >>> pt = np.zeros(128, dtype=np.uint8)
            >>> k = np.zeros(128, dtype=np.uint8)
            >>> ct = aes.encrypt(pt, k)
            >>> dec = aes.decrypt(ct, k)
            >>> np.array_equal(pt, dec)
            True
        """
        if ciphertext.size != 128:
            raise ValueError("密文必须为 128 位")
        if key.size not in (128, 192, 256):
            raise ValueError("密钥长度必须为 128/192/256 位")

        state = self._bytes_from_bits(ciphertext)
        round_keys = self._key_schedule(key)
        Nr = len(round_keys) - 1

        # 初始轮：加上最后一轮密钥
        state = self._add_round_key(state, round_keys[Nr])

        # 中间轮：逆行移位 → 逆字节替换 → 加轮密钥 → 逆列混淆
        for r in range(Nr - 1, 0, -1):
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, round_keys[r])
            state = self._inv_mix_columns(state)

        # 最后一轮：逆行移位 → 逆字节替换 → 加上首轮密钥
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)
        state = self._add_round_key(state, round_keys[0])

        return self._bits_from_bytes(state)


if __name__ == "__main__":
    aes = AES128()
    pt = np.zeros(128, dtype=np.uint8)
    k = np.zeros(128, dtype=np.uint8)
    ct = aes.encrypt(pt, k)
    dec = aes.decrypt(ct, k)
    print(np.array_equal(pt, dec))
