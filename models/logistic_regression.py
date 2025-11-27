from sklearn.linear_model import LogisticRegression
from utils.common_utils import load_config_yaml


config = load_config_yaml()


def buildLogisticRegression() -> LogisticRegression:
    """
    构建并返回逻辑回归分类器

    详细描述函数的功能和用途：
    使用 `liblinear` 求解器以适配小样本场景，设置较大的最大迭代次数与随机种子；
    适用于线性可分或近似线性可分的特征。

    Args:
        无

    Returns:
        LogisticRegression: 配置好的逻辑回归模型

    Raises:
        Exception: 无

    Example:
        >>> m = buildLogisticRegression()
        >>> hasattr(m, "fit") and hasattr(m, "predict")
        True
    """
    # 【新增代码说明】将逻辑回归模型拆分至独立文件，便于独立调参与测试
    return LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=config["RANDOM_STATE"])

