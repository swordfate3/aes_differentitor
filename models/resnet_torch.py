"""Torch 残差网络分类器构建

提供一个用于表格特征的轻量级残差网络封装，暴露与 scikit-learn 类似的
`fit/predict/predict_proba` 接口，便于与现有训练流程无缝集成。
"""


def buildResNet():
    """
    构建并返回基于 PyTorch 的残差网络分类器（二分类）

    详细描述函数的功能和用途：
    本函数在内部动态导入 PyTorch，并创建一个适用于表格数据的残差全连接网络。
    该模型提供与 scikit-learn 相似的接口（`fit`/`predict`/`predict_proba`），
    以兼容项目现有训练与评估代码。

    Args:
        无

    Returns:
        object: 具备 `fit/predict/predict_proba` 方法的分类器实例

    Raises:
        ImportError: 当未安装 `torch` 时抛出并提示安装

    Example:
        >>> # 需安装 torch 后使用
        >>> clf = buildResNet()
        >>> hasattr(clf, "fit") and hasattr(clf, "predict")
        True
    """
    # 【新增代码说明】动态引入 torch，避免在未安装环境下阻断整个项目
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except Exception as e:
        raise ImportError(
            "未检测到 PyTorch，请先安装：pip install torch"
        ) from e

    class ResidualBlock(nn.Module):
        """残差全连接块

        详细描述函数的功能和用途：
        两层线性层中间带 BN+ReLU，残差短接；输入输出维度一致。

        Args:
            dim (int): 特征维度（输入=输出）
            dropout (float): Dropout 比例

        Returns:
            nn.Module: 残差块模块
        """

        def __init__(self, dim: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.net(x)
            out = out + x
            return self.act(out)

    class TabularResNet(nn.Module):
        """用于表格特征的残差网络（二分类）"""

        def __init__(self, in_dim: int, hidden: int = 256, blocks: int = 4,
                     dropout: float = 0.1) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden, 2),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            logits = self.head(x)
            return logits

    class ResNetClassifier:
        """
        轻量封装的残差网络分类器（sklearn 风格接口）

        详细描述函数的功能和用途：
        - `fit(X, y)`: 训练网络，支持 numpy 数组输入
        - `predict(X)`: 返回类别标签（0/1）
        - `predict_proba(X)`: 返回类别概率（N×2）

        Args:
            epochs (int): 训练轮数，默认 20
            lr (float): 学习率，默认 1e-3
            batch_size (int): 批大小，默认 64
            hidden (int): 隐层维度，默认 256
            blocks (int): 残差块数量，默认 4
            dropout (float): Dropout 比例，默认 0.1

        Returns:
            ResNetClassifier: 分类器实例
        """

        def __init__(self, epochs: int = 20, lr: float = 1e-3, batch_size: int = 64,
                     hidden: int = 256, blocks: int = 4, dropout: float = 0.1) -> None:
            # 【新增代码说明】初始化训练超参数与占位属性
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.hidden = hidden
            self.blocks = blocks
            self.dropout = dropout
            self._model = None
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _ensure_model(self, in_dim: int) -> None:
            if self._model is None:
                self._model = TabularResNet(
                    in_dim, hidden=self.hidden, blocks=self.blocks, dropout=self.dropout
                ).to(self._device)

        def fit(self, X, y):
            """训练模型

            Args:
                X (np.ndarray): 特征矩阵，形状 (N, F)
                y (np.ndarray): 标签向量，形状 (N,)

            Returns:
                ResNetClassifier: 返回自身，便于链式调用
            """
            import numpy as np

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            self._ensure_model(X.shape[1])

            ds = TensorDataset(
                torch.from_numpy(X), torch.from_numpy(y)
            )
            dl = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

            self._model.train()
            for _ in range(self.epochs):
                for xb, yb in dl:
                    xb = xb.to(self._device)
                    yb = yb.to(self._device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
            return self

        def predict_proba(self, X):
            """预测类别概率

            Args:
                X (np.ndarray): 特征矩阵

            Returns:
                np.ndarray: 概率矩阵 (N, 2)
            """
            import numpy as np

            X = np.asarray(X, dtype=np.float32)
            if self._model is None:
                raise RuntimeError("模型尚未训练，请先调用 fit()")
            self._model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(X).to(self._device)
                logits = self._model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        def predict(self, X):
            """预测类别标签（0/1）"""
            import numpy as np

            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(np.int64)

    # 【新增代码说明】返回封装后的分类器实例
    return ResNetClassifier()

