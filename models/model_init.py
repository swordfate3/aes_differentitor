from models.svm import buildSvm
from models.random_forest import buildRandomForest
from models.logistic_regression import buildLogisticRegression
from models.naive_bayes import buildNaiveBayes
from models.mlp import buildMlp
from models.resnet_torch import buildResNet


def init_all_models():
    """
    初始化论文中的5种模型（分离至单文件构建）

    详细描述函数的功能和用途：
    调用分离后的各模型构建函数，返回名称到模型实例的映射；保持原有参数设置，
    便于训练、评估与交叉验证统一调用。

    Args:
        无

    Returns:
        dict[str, object]: 模型名称到已配置实例的字典

    Raises:
        Exception: 无

    Example:
        >>> ms = init_all_models()
        >>> set(ms.keys()) >= {"SVM", "Random Forest"}
        True
    """
    # 【修改代码说明】重构：由集中式定义改为调用各自文件中的构建函数
    models = {
        "SVM": buildSvm(),
        "Random Forest": buildRandomForest(),
        "Logistic Regression": buildLogisticRegression(),
        "Naive Bayes": buildNaiveBayes(),
        "MLP": buildMlp(),
        "ResNet": buildResNet(),
    }
    print("模型初始化完成：", list(models.keys()))
    return models
