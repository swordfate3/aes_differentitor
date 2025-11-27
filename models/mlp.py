from sklearn.neural_network import MLPClassifier
from utils.common_utils import load_config_yaml


config = load_config_yaml()


def buildMlp() -> MLPClassifier:
    """
    构建并返回多层感知机分类器（MLP）

    详细描述函数的功能和用途：
    三层每层 1024 神经元，ReLU 激活，Adam 优化，关闭 early_stopping 以避免
    小样本验证集导致的类别不足问题；随机种子来自配置，保证可复现。

    Args:
        无

    Returns:
        MLPClassifier: 配置好的 MLP 模型

    Raises:
        Exception: 无

    Example:
        >>> m = buildMlp()
        >>> hasattr(m, "fit") and hasattr(m, "predict")
        True
    """
    # 【新增代码说明】将 MLP 模型拆分至独立文件，便于独立调参
    return MLPClassifier(
        hidden_layer_sizes=(1024, 1024, 1024),
        activation="relu",
        solver="adam",
        max_iter=200,
        batch_size=32,
        random_state=config["RANDOM_STATE"],
        early_stopping=False,
    )

