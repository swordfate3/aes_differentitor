import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from models.model_init import init_all_models
from utils.common_utils import load_config_yaml

config = load_config_yaml()
TEST_SIZE = config["TEST_SIZE"]
RANDOM_STATE = config["RANDOM_STATE"]
KFOLD_SPLITS = config["KFOLD_SPLITS"]


def train_models(models, X_train, y_train, selected_names=None):
    """
    训练模型（支持按名称选择）

    详细描述函数的功能和用途：
    接收模型字典与训练数据；若提供 `selected_names`，仅训练指定名称的模型，
    否则训练所有模型。名称匹配不区分大小写。

    Args:
        models (dict): 名称到模型实例的映射
        X_train (np.ndarray): 训练集特征
        y_train (np.ndarray): 训练集标签
        selected_names (list[str] | None): 指定训练的模型名称列表

    Returns:
        dict: 已训练的模型字典

    Raises:
        Exception: 无

    Example:
        >>> import numpy as np
        >>> ms = init_all_models()
        >>> X = np.random.randn(20, 4); y = np.random.randint(0, 2, 20)
        >>> trained = train_models(ms, X, y, ["SVM", "Logistic Regression"])
        >>> set(trained.keys()) == {"SVM", "Logistic Regression"}
        True
    """
    trained_models = {}
    if selected_names:
        want = {n.strip().lower() for n in selected_names}
        items = [(name, model) for name, model in models.items() if name.lower() in want]
        missing = want - {name.lower() for name, _ in items}
        if missing:
            print(f"警告：以下模型名称未找到，将忽略：{sorted(list(missing))}")
    else:
        items = list(models.items())

    for name, model in items:
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """评估模型（输出分类报告）"""
    eval_results = {}
    print("\n=== 模型测试集评估结果 ===")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        # 生成分类报告
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Random", "Cipher"],
            output_dict=True,
            digits=4,
        )
        eval_results[name] = report
        # 打印报告
        print(f"\n{name}:")
        report_df = pd.DataFrame(report).T[
            ["precision", "recall", "f1-score", "support"]
        ]
        print(report_df.round(4))
    return eval_results


def k_fold_cross_validation(X, y, selected_names=None):
    """
    k折交叉验证（k=5/10，支持按名称选择）

    详细描述函数的功能和用途：
    对指定或全部模型执行 k 折交叉验证；当提供 `selected_names` 时仅验证这些模型。

    Args:
        X (np.ndarray): 全量特征数据
        y (np.ndarray): 标签数据
        selected_names (list[str] | None): 指定参与交叉验证的模型名称列表

    Returns:
        None: 仅打印结果

    Raises:
        Exception: 无

    Example:
        >>> import numpy as np
        >>> X = np.random.randn(40, 5); y = np.random.randint(0, 2, 40)
        >>> k_fold_cross_validation(X, y, ["SVM"])  # 仅验证 SVM
    """
    models = init_all_models()
    print("\n=== k折交叉验证结果 ===")
    for k in KFOLD_SPLITS:
        print(f"\n{k}-折验证：")
        kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        names = list(models.keys())
        if selected_names:
            want = {n.strip().lower() for n in selected_names}
            names = [n for n in names if n.lower() in want]
            if not names:
                print("警告：指定模型名称在当前集合中均未找到，使用全部模型进行验证")
                names = list(models.keys())
        for name in names:
            precisions, recalls, f1s = [], [], []
            for train_idx, val_idx in kf.split(X):
                X_kf_train, X_kf_val = X[train_idx], X[val_idx]
                y_kf_train, y_kf_val = y[train_idx], y[val_idx]
                # 重新初始化模型（避免跨折污染）
                model = init_all_models()[name]
                model.fit(X_kf_train, y_kf_train)
                y_pred = model.predict(X_kf_val)
                # 计算指标
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_kf_val, y_pred, average="binary"
                )
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
            # 输出平均结果
            avg_prec = np.mean(precisions)
            avg_rec = np.mean(recalls)
            avg_f1 = np.mean(f1s)
            print(
                f"{name} - 精度：{avg_prec:.4f}, 召回：{avg_rec:.4f}, F1：{avg_f1:.4f}"
            )


def split_train_test(X, y):
    """划分训练/测试集（分层抽样）"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y  # 保证标签平衡
    )
    print(f"数据划分完成：训练集{X_train.shape}, 测试集{X_test.shape}")
    return X_train, X_test, y_train, y_test
