from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from utils.common_utils import load_config_yaml

config = load_config_yaml()
RANDOM_STATE = config["RANDOM_STATE"]


def init_all_models():
    """初始化论文中的5种模型（参数严格对齐）"""
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
        ),  # 删除代码：移除不支持的 random_state 参数，避免 TypeError
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,  # 多核CPU加速
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(1024, 1024, 1024),  # 3层1024神经元
            activation="relu",
            solver="adam",
            max_iter=200,
            batch_size=32,
            random_state=RANDOM_STATE,
            early_stopping=False,
        ),  # 修改代码：关闭early_stopping，避免小样本验证集类别不足错误
    }
    print("模型初始化完成：", list(models.keys()))
    return models
