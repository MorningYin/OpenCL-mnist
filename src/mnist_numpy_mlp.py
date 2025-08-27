"""
纯 NumPy 实现的 MLP（用于训练并导出参数给 C++ OpenCL 项目）：
- 下载/解析 MNIST 数据集（idx 格式）
- 构建 784 -> hidden(s) -> 10 的前馈网络（Linear + ReLU）
- 训练后将每层权重、偏置以 .txt 文本格式导出（供 C++ 读取）

导出格式：
- 权重矩阵 linear{i}_W.txt：首行 "rows cols"，随后按行主序输出浮点
- 偏置向量 linear{i}_b.txt：首行 "len 1"，随后每行一个浮点
"""

import argparse
import gzip
import os
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import urllib.request


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, dst_path: Path) -> None:
    """若目标文件不存在则下载到本地。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    print(f"Downloading {url} -> {dst_path} ...")
    with urllib.request.urlopen(url) as response, open(dst_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"Saved {dst_path}")


def ensure_mnist_downloaded(data_dir: Path) -> dict:
    """确保 MNIST 四个 .gz 文件已下载到 data_dir，返回路径字典。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        k: data_dir / Path(MNIST_URLS[k]).name for k in MNIST_URLS
    }
    for k, url in MNIST_URLS.items():
        _download_file(url, paths[k])
    return paths


def _parse_idx_images(gz_path: Path) -> np.ndarray:
    """解析 idx3-ubyte.gz 图像文件，返回形状 (N, 784) 且归一化到 [0,1] 的 np.float32。"""
    with gzip.open(gz_path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(
            ">IIII", f.read(16)
        )
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic}")
        buf = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, num_rows * num_cols).astype(np.float32) / 255.0
        return data


def _parse_idx_labels(gz_path: Path) -> np.ndarray:
    """解析 idx1-ubyte.gz 标签文件，返回形状 (N,) 的 np.uint8 -> np.int64。"""
    with gzip.open(gz_path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic}")
        buf = f.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def load_mnist(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """下载并载入 MNIST，返回 (x_train, y_train, x_test, y_test)。"""
    paths = ensure_mnist_downloaded(data_dir)
    x_train = _parse_idx_images(paths["train_images"])  # (60000, 784)
    y_train = _parse_idx_labels(paths["train_labels"])  # (60000,)
    x_test = _parse_idx_images(paths["test_images"])    # (10000, 784)
    y_test = _parse_idx_labels(paths["test_labels"])    # (10000,)
    return x_train, y_train, x_test, y_test


class Linear:
    """线性层：out = x @ W + b，包含反向与简单 SGD 更新。"""
    def __init__(self, in_features: int, out_features: int, rng: np.random.RandomState):
        # He initialization for ReLU
        std = np.sqrt(2.0 / in_features)
        self.weights = rng.randn(in_features, out_features).astype(np.float32) * std
        self.bias = np.zeros((out_features,), dtype=np.float32)
        # cache for backward
        self._input = None

        # gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向：保存输入以便反向使用。"""
        self._input = x  # (N, in)
        return x @ self.weights + self.bias  # (N, out)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向：计算对权重/偏置/输入的梯度。"""
        # grad_output: (N, out)
        # gradients w.r.t weights and bias
        self.grad_weights[...] = self._input.T @ grad_output
        self.grad_bias[...] = grad_output.sum(axis=0)
        # gradient w.r.t input
        grad_input = grad_output @ self.weights.T
        return grad_input

    def sgd_step(self, lr: float, batch_size: int) -> None:
        """按批平均后进行一次 SGD 更新。"""
        # average gradients by batch size
        self.weights -= lr * (self.grad_weights / batch_size)
        self.bias -= lr * (self.grad_bias / batch_size)


class ReLU:
    """ReLU 激活：y = max(0, x)，保存掩码以便反向传播。"""
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向：记录正值掩码并置负值为 0。"""
        self._mask = x > 0
        out = x.copy()
        out[~self._mask] = 0
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向：仅对正值位置传递梯度。"""
        grad = grad_output.copy()
        grad[~self._mask] = 0
        return grad


class MLP:
    """多层感知机：Linear/（ReLU）*，末层不接 ReLU。"""
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], rng))
            if i < len(dims) - 2:  # no ReLU after last linear
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        """顺序前向计算。"""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> None:
        """反向传播通过所有层。"""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr: float, batch_size: int) -> None:
        """对所有线性层执行一次 SGD 更新。"""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.sgd_step(lr, batch_size)

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """按批预测 argmax。"""
        preds = []
        for i in range(0, x.shape[0], batch_size):
            logits = self.forward(x[i : i + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds, axis=0)

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回所有线性层的 (W, b) 列表。"""
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append((layer.weights, layer.bias))
        return params


def softmax_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """交叉熵损失 + 对 logits 的梯度。"""
    # logits: (N, C), labels: (N,) integer class ids
    # returns (loss_scalar, grad_logits)
    max_logits = logits.max(axis=1, keepdims=True)
    stabilized = logits - max_logits
    exp = np.exp(stabilized)
    probs = exp / exp.sum(axis=1, keepdims=True)

    N = logits.shape[0]
    # gather correct class probabilities
    correct_logprobs = -np.log(probs[np.arange(N), labels] + 1e-12)
    loss = float(correct_logprobs.mean())

    grad = probs
    grad[np.arange(N), labels] -= 1.0
    # do not divide by N here; handled in optimizer step to be consistent with layer grads
    return loss, grad


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.RandomState):
    """打乱索引后按批产出 (xb, yb)。"""
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """计算分类准确率。"""
    return float((pred == target).mean())


def save_parameters_txt(model: MLP, out_dir: Path) -> None:
    """按项目读取格式导出线性层参数到文本文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # save meta
    meta_path = out_dir / "model_meta.txt"
    dims = []
    for layer in model.layers:
        if isinstance(layer, Linear):
            dims.append((layer.weights.shape[0], layer.weights.shape[1]))
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("# Layer shapes (in out) per Linear in order\n")
        for i, (din, dout) in enumerate(dims):
            f.write(f"linear_{i}: {din} {dout}\n")

    # save each linear weight and bias
    lin_idx = 0
    for layer in model.layers:
        if not isinstance(layer, Linear):
            continue
        w_path = out_dir / f"linear{lin_idx}_W.txt"
        b_path = out_dir / f"linear{lin_idx}_b.txt"

        W = layer.weights
        b = layer.bias
        with open(w_path, "w", encoding="utf-8") as wf:
            wf.write(f"{W.shape[0]} {W.shape[1]}\n")
            for r in range(W.shape[0]):
                row_str = " ".join(f"{v:.6f}" for v in W[r])
                wf.write(row_str + "\n")

        with open(b_path, "w", encoding="utf-8") as bf:
            bf.write(f"{b.shape[0]} 1\n")
            for v in b:
                bf.write(f"{v:.6f}\n")

        lin_idx += 1

    print(f"Parameters saved to: {out_dir}")


def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_sizes: List[int],
    sample_size: int,
    output_dir: Path,
    seed: int,
    data_dir: Path,
):
    """在 MNIST 上训练简单的 MLP，并导出参数到 out_dir。"""
    rng = np.random.RandomState(seed)
    x_train, y_train, x_test, y_test = load_mnist(data_dir)

    if sample_size is not None and sample_size > 0:
        sample_size = min(sample_size, x_train.shape[0])
        idx = rng.choice(x_train.shape[0], size=sample_size, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    input_dim = x_train.shape[1]
    num_classes = int(y_train.max()) + 1
    model = MLP(input_dim=input_dim, hidden_dims=hidden_sizes, num_classes=num_classes, seed=seed)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for xb, yb in iterate_minibatches(x_train, y_train, batch_size, rng):
            logits = model.forward(xb)
            loss, grad_logits = softmax_cross_entropy_with_logits(logits, yb)

            # backward
            model.backward(grad_logits)
            # update
            model.step(learning_rate, batch_size=xb.shape[0])

            running_loss += loss
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        # evaluate
        test_pred = model.predict(x_test)
        test_acc = accuracy(test_pred, y_test)
        print(f"Epoch {epoch:02d}/{epochs} - loss: {avg_loss:.4f} - test_acc: {test_acc:.4f}")

    save_parameters_txt(model, output_dir)


def parse_hidden_sizes(s: str) -> List[int]:
    """解析形如 '128,64' 或 '256;128;64' 的隐藏层配置字符串。"""
    s = s.strip()
    if not s:
        return []
    parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    """命令行入口：训练并保存参数。"""
    parser = argparse.ArgumentParser(
        description="Pure NumPy MLP for MNIST classification. Saves parameters to .txt for C++"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--hidden-sizes", type=str, default="128,64", help="e.g. '256,128,64'"
    )
    parser.add_argument(
        "--sample-size", type=int, default=0, help="train on subset for quick runs (0=full)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="model_out", help="directory to save .txt params"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path("data") / "mnist"),
        help="where to download/store MNIST gz files",
    )

    args = parser.parse_args()

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    if len(hidden_sizes) == 0:
        hidden_sizes = [128, 64]

    sample_size = int(args.sample_size) if args.sample_size and args.sample_size > 0 else 0
    out_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    train(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        hidden_sizes=hidden_sizes,
        sample_size=sample_size,
        output_dir=out_dir,
        seed=int(args.seed),
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()


