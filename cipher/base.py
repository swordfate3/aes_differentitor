"""通用分组密码实现基类

提供统一的加解密接口，适用于所有分组密码算法。仅负责原始分组数据的
加解密，不涉及填充、模式（如 CBC/CTR）或 IV 等其他因素。具体算法
实现需在子类中提供底层分组加密器。
"""

from typing import Any, Tuple


class BaseCipher:
    """通用分组密码基类

    仅提供原始分组数据的加解密接口：输入数据长度必须是分组大小的整数倍。
    子类通过实现 `_createCipher` 返回具备 `encrypt/decrypt` 方法的底层加密器。

    Args:
        key (bytes): 密钥字节串
        block_size (int): 分组大小（字节），例如 AES 为 16

    Raises:
        ValueError: 当密钥长度或数据长度非法时抛出

    Example:
        >>> # base = cipherBase(key=b"\x00" * 16, block_size=16)
        >>> # 仅用于说明，需在子类中实现 _createCipher
    """

    def __init__(self, key: bytes, block_size: int) -> None:
        """初始化基础加密器

        Args:
            key (bytes): 密钥字节串
            block_size (int): 分组大小（字节）

        Raises:
            TypeError: 当密钥类型不是 bytes 时抛出
            ValueError: 当分组大小不合法时抛出

        Example:
            >>> BaseCipher(b"\x00" * 16, 16)  # 仅用于子类继承场景
        """
        # 新增代码：类型与范围校验，保证健壮性
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("密钥必须为 bytes 或 bytearray")
        if int(block_size) <= 0:
            raise ValueError("分组大小必须为正整数")
        # 新增代码：保存密钥与分组大小
        self._key = bytes(key)
        self._block_size = int(block_size)

    # 新增代码：密钥长度校验工具（比特）
    def validateKeyForLengths(self, allowed_lengths_bits: Tuple[int, ...]) -> None:
        """校验密钥长度是否匹配允许的比特数

        Args:
            allowed_lengths_bits (Tuple[int, ...]): 允许的密钥长度（比特）

        Raises:
            ValueError: 当密钥长度不在允许范围内时抛出

        Example:
            >>> base = BaseCipher(b"\x00" * 16, 16)
            >>> base.validateKeyForLengths((128,))
        """
        key_bits = len(self._key) * 8
        if key_bits not in allowed_lengths_bits:
            raise ValueError(
                f"非法密钥长度: {key_bits}，允许: {allowed_lengths_bits}"
            )

    # 新增代码：原始分组加密（不进行填充与 IV 处理）
    def encrypt(self, data: bytes) -> bytes:
        """加密原始分组数据

        Args:
            data (bytes): 待加密数据，长度必须是 `block_size` 的整数倍

        Returns:
            bytes: 密文字节串

        Raises:
            ValueError: 当数据长度不是分组大小的整数倍时抛出

        Example:
            >>> # 子类实例中使用：cipher.encrypt(b"\x00" * 16)
        """
        # 新增代码：输入类型校验
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("输入数据必须为 bytes 或 bytearray")
        if len(data) % self._block_size != 0:
            raise ValueError("数据长度必须是分组大小的整数倍")
        cipher = self._createCipher()
        return cipher.encrypt(bytes(data))

    # 新增代码：原始分组解密（不进行填充与 IV 处理）
    def decrypt(self, data: bytes) -> bytes:
        """解密原始分组数据

        Args:
            data (bytes): 待解密数据，长度必须是 `block_size` 的整数倍

        Returns:
            bytes: 明文字节串

        Raises:
            ValueError: 当数据长度不是分组大小的整数倍时抛出

        Example:
            >>> # 子类实例中使用：cipher.decrypt(b"\x00" * 16)
        """
        # 新增代码：输入类型校验
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("输入数据必须为 bytes 或 bytearray")
        if len(data) % self._block_size != 0:
            raise ValueError("数据长度必须是分组大小的整数倍")
        cipher = self._createCipher()
        return cipher.decrypt(bytes(data))

    # 新增代码：抽象工厂，由子类提供具体分组加密器
    def _createCipher(self) -> Any:  # pragma: no cover
        """创建具体分组加密器

        子类必须实现该方法以返回底层加密器对象（需具备 encrypt/decrypt）。

        Returns:
            Any: 底层加密器对象

        Raises:
            NotImplementedError: 基类不提供具体实现
        """
        raise NotImplementedError

    # 新增代码：只读属性访问，便于外部获取配置
    @property
    def blockSize(self) -> int:
        """获取分组大小（字节）

        Returns:
            int: 分组大小

        Example:
            >>> BaseCipher(b"\x00" * 16, 16).blockSize
            16
        """
        return self._block_size

    @property
    def keyBytes(self) -> bytes:
        """获取密钥字节串

        Returns:
            bytes: 密钥

        Example:
            >>> BaseCipher(b"\x01" * 16, 16).keyBytes[:1] == b"\x01"
            True
        """
        return self._key
