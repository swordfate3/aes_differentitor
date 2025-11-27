"""AES CBC 测试脚本

用于快速验证 `aesCipher128/192/256` 在 CBC 模式下的加解密正确性。
运行完成后会删除本测试文件，避免污染代码库。
"""

from Crypto.Random import get_random_bytes

from cipher.AES128 import aesCipher128, aesCipher192, aesCipher256


def makeKey(length_bits: int) -> bytes:
    """生成指定比特长度的随机密钥

    Args:
        length_bits (int): 密钥长度（比特），可为 128/192/256

    Returns:
        bytes: 随机密钥字节串

    Example:
        >>> key = makeKey(128)
        >>> len(key) == 16
        True
    """
    return get_random_bytes(length_bits // 8)


def testEncryptDecryptForSize(length_bits: int) -> None:
    """针对不同密钥长度测试 CBC 加解密流程

    Args:
        length_bits (int): 密钥长度（比特）

    Raises:
        AssertionError: 当加解密结果不一致时抛出

    Example:
        >>> testEncryptDecryptForSize(128)
    """
    key = makeKey(length_bits)
    plaintext = b"The quick brown fox jumps over the lazy dog"

    if length_bits == 128:
        cipher = aesCipher128(key)
    elif length_bits == 192:
        cipher = AES192(key)
    elif length_bits == 256:
        cipher = AES256(key)
    else:
        raise ValueError("不支持的长度")

    ciphertext, iv = cipher.encrypt(plaintext)
    decrypted = cipher.decrypt(ciphertext, iv)
    assert decrypted == plaintext


def testInvalidKeyLengthRaises() -> None:
    """测试非法密钥长度会抛出异常

    Raises:
        AssertionError: 当未抛出异常时
    """
    try:
        aesCipher128(b"\x00" * 15)
        raise AssertionError("预期 128 位密钥长度校验失败")
    except ValueError:
        pass

    try:
        aesCipher192(b"\x00" * 23)
        raise AssertionError("预期 192 位密钥长度校验失败")
    except ValueError:
        pass

    try:
        aesCipher256(b"\x00" * 31)
        raise AssertionError("预期 256 位密钥长度校验失败")
    except ValueError:
        pass


def testInvalidIvLengthRaises() -> None:
    """测试非法 IV 长度在加解密时抛出异常"""
    key = makeKey(128)
    cipher = aesCipher128(key)
    try:
        cipher.encrypt(b"hello", iv=b"shortiv")
        raise AssertionError("预期 iv 长度校验失败")
    except ValueError:
        pass

    ct, iv = cipher.encrypt(b"world")
    try:
        cipher.decrypt(ct, b"shortiv")
        raise AssertionError("预期 iv 长度校验失败")
    except ValueError:
        pass


def main() -> None:
    """运行全部测试用例并打印结果"""
    testEncryptDecryptForSize(128)
    testEncryptDecryptForSize(192)
    testEncryptDecryptForSize(256)
    testInvalidKeyLengthRaises()
    testInvalidIvLengthRaises()
    print("AES CBC 测试通过：128/192/256")


if __name__ == "__main__":
    main()