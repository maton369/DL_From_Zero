# coding: utf-8
import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import psutil

# Jupyter Notebook 対応 (__file__ を使用しない)
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from dataset.mnist import load_mnist

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    else:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


# バッチサイズの候補
batch_sizes = [1, 10, 100, 1000, 10000, 20000, 50000, 100000]

x, t = get_data()

# 結果保存用
results = []

print(
    f"{'BatchSize':>10} | {'Acc(MEAN)':>9} | {'Acc(STD)':>9} | {'Time(MEAN)':>10} | {'Time(STD)':>9} | {'Memory(MB)':>10}"
)
print("-" * 75)

for batch_size in batch_sizes:
    accuracies = []
    times = []
    memory_usages = []

    for _ in range(3):  # 同じバッチサイズで3回測定
        network = init_network()
        process = psutil.Process(os.getpid())  # 現在のプロセスを取得
        accuracy_cnt = 0
        start_time = time.time()

        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == t[i : i + batch_size])

        elapsed = time.time() - start_time
        accuracy = accuracy_cnt / len(x)
        mem_usage = process.memory_info().rss / (1024 * 1024)  # MB

        accuracies.append(accuracy)
        times.append(elapsed)
        memory_usages.append(mem_usage)

    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    time_mean = np.mean(times)
    time_std = np.std(times)
    mem_mean = np.mean(memory_usages)

    print(
        f"{batch_size:>10} | {acc_mean:.4f}   | {acc_std:.4f}   | {time_mean:>10.4f} | {time_std:.4f} | {mem_mean:>10.2f}"
    )

    results.append((batch_size, acc_mean, acc_std, time_mean, time_std, mem_mean))

# グラフ描画（平均値のみ）
batch_list = [r[0] for r in results]
acc_means = [r[1] for r in results]
time_means = [r[3] for r in results]

fig, ax1 = plt.subplots()

ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Accuracy", color="tab:red")
ax1.plot(batch_list, acc_means, marker="o", color="tab:red", label="Accuracy")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.set_xscale("log")
ax1.set_xticks(batch_list)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax2 = ax1.twinx()
ax2.set_ylabel("Time (s)", color="tab:blue")
ax2.plot(
    batch_list,
    time_means,
    marker="s",
    linestyle="--",
    color="tab:blue",
    label="Time (s)",
)
ax2.tick_params(axis="y", labelcolor="tab:blue")

# メモリ使用量グラフの描画
mem_means = [r[5] for r in results]

plt.figure()
plt.plot(
    batch_list, mem_means, marker="^", color="tab:green", label="Memory Usage (MB)"
)
plt.xscale("log")
plt.xlabel("Batch Size")
plt.ylabel("Memory Usage (MB)")
plt.title("Batch Size vs Memory Usage")
plt.grid(True)
plt.savefig("batch_memory_usage.png")
plt.show()
