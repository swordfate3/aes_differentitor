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


def train_models(models, X_train, y_train):
    """训练所有模型"""
    trained_models = {}
    for name, model in models.items():
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


def k_fold_cross_validation(X, y):
    """k折交叉验证（k=5/10）"""
    models = init_all_models()
    print("\n=== k折交叉验证结果 ===")
    for k in KFOLD_SPLITS:
        print(f"\n{k}-折验证：")
        kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        for name in models.keys():
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
