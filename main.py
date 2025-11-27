import argparse  # 新增代码：引入命令行参数解析
from utils.data_generate_utils import generate_and_save_data, load_data
from features.feature_process import (
    extract_all_features,
    filter_features,
    prepare_train_data,
)
from models.model_init import init_all_models
from trainer.train_model import (
    split_train_test,
    train_models,
    evaluate_models,
    k_fold_cross_validation,
)
from visualization.plots import plot_pca_features, plot_top_kde_features, plot_tsne_features
from utils.common_utils import load_config_yaml

# 加载配置
config = load_config_yaml()


def parseArgs():
    """解析命令行参数

    Returns:
        argparse.Namespace: 参数命名空间，包含步骤选择信息

    Example:
        >>> # 默认运行所有步骤
        >>> args = parseArgs()
        >>> # 通过 --step 指定多个步骤
        >>> # python main.py --step features filter
    """
    parser = argparse.ArgumentParser(description="控制实验主流程的执行步骤")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help=(
            "选择执行步骤，逗号分隔。支持: all,data,features,filter,prepare,train,cv,plot；"
            "例如: --steps data,features,filter"
        ),
    )
    parser.add_argument(
        "--step",
        type=str,
        nargs="+",
        help=(
            "按空格分隔指定多个步骤，可重复。支持: all data features filter prepare train cv plot；"
            "例如: --step features filter"
        ),
    )
    parser.add_argument(
        "--cipher-key",
        "--cipher_key",
        dest="cipher_key",
        type=str,
        default=None,
        help=(
            "选择密文数据集键，例如: aes128_custom_single, aes128_custom_multi, aes256_single；"
            "不指定时自动优先选择自定义AES单密钥"
        ),
    )
    return parser.parse_args()


def main():
    """实验全流程主函数（支持步骤选择）

    根据命令行 `--steps` 控制执行哪些步骤，默认 `all`（全步骤）。当选择后续步
    骤但依赖的前序结果不存在时，会自动补齐必要步骤以保证流程可运行。

    Raises:
        FileNotFoundError: 当数据集缺失且生成失败时抛出

    Example:
        >>> # 默认运行全步骤
        >>> main()
        >>> # 仅生成数据与特征提取
        >>> # python main.py --steps data,features
    """
    args = parseArgs()  # 新增代码：读取命令行步骤选择
    if getattr(args, "step", None):
        steps = set(s.strip() for s in args.step)
    elif getattr(args, "steps", None):
        steps = set(s.strip() for s in args.steps.split(","))
    else:
        steps = {"all"}
    run_all = "all" in steps

    print("=" * 50)
    print("FESLA实验流程启动（全轮AES-256唯密文区分）")  # 修改代码：与论文目标一致
    print("=" * 50)

    data = None
    cipher_features = None
    random_features = None
    valid_indices = None
    X = y = None
    X_train = X_test = y_train = y_test = None
    trained_models = None
    cipher_key = None

    # 步骤1：生成/加载数据
    if run_all or "data" in steps:
        try:
            print("\n步骤1/6：加载数据集...")
            data = load_data()
            print(f"数据集特征{data.keys()}")
            print(f"数据集样本形状{data['aes128_custom_single'].shape}")
            print(f"随机数据集样本形状{data['random_datasets'].shape}")
            print(f"随机数据集样本前5{data['aes128_custom_single'][0][:5]}")
        except FileNotFoundError:
            print("\n步骤1/6：数据集未找到，开始生成...")
            generate_and_save_data()
            data = load_data()
            print(f"数据集特征{data.keys()}")
    else:
        # 新增代码：若未选择 data 步骤，但后续需要数据，则尝试加载
        try:
            data = load_data()
        except FileNotFoundError:
            print("数据集缺失，为保证后续步骤执行，自动生成数据...")
            generate_and_save_data()
            data = load_data()

    # 基于参数/数据选择密文数据集键
    if data is not None:
        if getattr(args, "cipher_key", None) and args.cipher_key in data:
            cipher_key = args.cipher_key
        else:
            cipher_key = (
                "aes128_custom_single"
                if "aes128_custom_single" in data
                else ("aes256_single" if "aes256_single" in data else "aes128_single")
            )
        print(f"选择密文数据集{ cipher_key}")

    # 步骤2：提取特征
    if run_all or "features" in steps:
        print("\n步骤2/6：提取统计特征...")
        print(f"选择密文数据集{ cipher_key}")
        cipher_batches = data[cipher_key]
        random_datasets = data["random_datasets"]
        cipher_features, random_features = extract_all_features(
            cipher_batches, random_datasets
        )
        print("特征提取完成，密文特征形状:", cipher_features.shape)
        print(f"密文特征前5{cipher_features[0][:5]}")
        print("随机特征形状:", random_features.shape)
        print(f"随机特征前5{random_features[0][:5]}")

    # 步骤3：筛选特征
    if run_all or "filter" in steps:
        if cipher_features is None or random_features is None:
            print("未显式选择提取特征，自动进行提取以支持筛选...")
            print(f"选择密文数据集{ cipher_key}")       
            cipher_batches = data[cipher_key]
            random_datasets = data["random_datasets"]
            cipher_features, random_features = extract_all_features(
                cipher_batches, random_datasets
            )
        print("\n步骤3/6：筛选有效特征...")
        valid_indices = filter_features(cipher_features, random_features)
        print(f"筛选有效特征索引{valid_indices[:5]}")

    # 步骤4：准备训练数据
    if run_all or "prepare" in steps:
        if valid_indices is None:
            print("未显式选择筛选特征，自动进行筛选以支持准备数据...")
            if cipher_features is None or random_features is None:
                print(f"选择密文数据集{ cipher_key}")       
                cipher_batches = data[cipher_key]
                random_datasets = data["random_datasets"]
                cipher_features, random_features = extract_all_features(
                    cipher_batches, random_datasets
                )
            valid_indices = filter_features(cipher_features, random_features)
        print("\n步骤4/6：准备训练/测试数据...")
        X, y = prepare_train_data(cipher_features, random_features, valid_indices)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        print(f"训练集样本形状{X_train.shape}")
        print(f"测试集样本形状{X_test.shape}")
        print(f"前5训练样本{X_train[:5]}")

    # 步骤5：模型训练与评估
    if run_all or "train" in steps:
        if X_train is None:
            print("未显式选择准备数据，自动进行数据准备以支持训练...")
            if valid_indices is None:
                if cipher_features is None or random_features is None:
                    print(f"选择密文数据集{ cipher_key}")           
                    cipher_batches = data[cipher_key]
                    random_datasets = data["random_datasets"]
                    cipher_features, random_features = extract_all_features(
                        cipher_batches, random_datasets
                    )
                # valid_indices = filter_features(cipher_features, random_features)
            X, y = prepare_train_data(cipher_features, random_features, valid_indices)
            print(f"训练数据样本形状{X.shape}")
            X_train, X_test, y_train, y_test = split_train_test(X, y)
        print("\n步骤5/6：模型训练与评估...")
        models = init_all_models()
        trained_models = train_models(models, X_train, y_train)
        evaluate_models(trained_models, X_test, y_test)

    # 步骤6：交叉验证与可视化
    if run_all or "cv" in steps or "plot" in steps:
        if X is None or valid_indices is None:
            print("为支持交叉验证/可视化，自动补齐前置步骤...")
            if cipher_features is None or random_features is None:
                print(f"选择密文数据集{ cipher_key}")           
                cipher_batches = data[cipher_key]
                random_datasets = data["random_datasets"]
                cipher_features, random_features = extract_all_features(
                    cipher_batches, random_datasets
                )
            if valid_indices is None:
                valid_indices = filter_features(cipher_features, random_features)
            X, y = prepare_train_data(cipher_features, random_features, valid_indices)
        if run_all or "cv" in steps:
            print("\n步骤6A：交叉验证...")
            k_fold_cross_validation(X, y)
        if run_all or "plot" in steps:
            print("\n步骤6B：可视化（PCA + tSNE + Top3 KDE）...")
            X_cipher = cipher_features[:, valid_indices]
            X_random = random_features[:, valid_indices]
            plot_pca_features(X_cipher, X_random, valid_indices)
            plot_tsne_features(X_cipher, X_random, valid_indices)
            if trained_models is None:
                models = init_all_models()
                X_train, X_test, y_train, y_test = split_train_test(X, y)
                trained_models = train_models(models, X_train, y_train)
            plot_top_kde_features(
                X_cipher, X_random, valid_indices, trained_models["Random Forest"]
            )

    print("\n" + "=" * 50)
    print("实验流程完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
