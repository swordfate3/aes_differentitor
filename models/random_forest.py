from sklearn.ensemble import RandomForestClassifier
from utils.common_utils import load_config_yaml


config = load_config_yaml()


def buildRandomForest() -> RandomForestClassifier:
    """
    构建并返回随机森林分类器

    详细描述函数的功能和用途：
    使用 100 棵树、默认最大深度与最小分裂数，设置随机种子与多核加速；
    适合中小规模特征的稳健二分类。

    Args:
        无

    Returns:
        RandomForestClassifier: 配置好的随机森林模型

    Raises:
        Exception: 无

    Example:
        >>> m = buildRandomForest()
        >>> hasattr(m, "fit") and hasattr(m, "predict")
        True
    """
    # 【新增代码说明】将随机森林模型拆分至独立文件，参数与旧实现一致
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=config["RANDOM_STATE"],
        n_jobs=-1,
    )

