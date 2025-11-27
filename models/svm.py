from sklearn.svm import SVC


def buildSvm() -> SVC:
    """
    构建并返回 SVM 模型（RBF 内核）

    详细描述函数的功能和用途：
    使用 RBF 核的支持向量机，启用概率输出，适用于二分类任务；
    超参数与原始实现保持一致，便于与论文结果对齐。

    Args:
        无

    Returns:
        SVC: 配置好的 SVM 分类器实例

    Raises:
        Exception: 无

    Example:
        >>> m = buildSvm()
        >>> hasattr(m, "fit") and hasattr(m, "predict")
        True
    """
    # 【新增代码说明】将原先集中定义的 SVM 模型拆分至独立文件，便于维护与扩展
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)

