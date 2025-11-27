from sklearn.naive_bayes import GaussianNB


def buildNaiveBayes() -> GaussianNB:
    """
    构建并返回朴素贝叶斯分类器（高斯）

    详细描述函数的功能和用途：
    高斯朴素贝叶斯适用于连续特征，训练速度快，作为轻量基线模型。

    Args:
        无

    Returns:
        GaussianNB: 配置好的朴素贝叶斯模型

    Raises:
        Exception: 无

    Example:
        >>> m = buildNaiveBayes()
        >>> hasattr(m, "fit") and hasattr(m, "predict")
        True
    """
    # 【新增代码说明】将朴素贝叶斯模型拆分至独立文件
    return GaussianNB()

