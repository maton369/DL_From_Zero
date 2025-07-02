import numpy as np
import time
import matplotlib.pyplot as plt
import os

# ====== 固定乱数シードを設定 ======
np.random.seed(42)  # 例: 固定値42を指定

# ====== Row-major (C) vs Column-major (F) memory layout check ======

print("== Row-major vs Column-major ==")
a = np.random.randn(100, 100)

b = np.array(a, order="C")  # Row-major
c = np.array(a, order="F")  # Column-major

print("b.strides (C order):", b.strides)
print("c.strides (F order):", c.strides)
print("Arrays are equal:", np.allclose(b, c))

# ====== Benchmark: contiguous vs non-contiguous view ======

print("\n== Performance comparison by stride ==")

x = np.ones((100_000,), dtype=np.float64)
y = np.ones((100_000 * 100,), dtype=np.float64)[::100]
y_copy = np.copy(y)

print("x.strides:", x.strides)
print("y.strides:", y.strides)
print("y_copy.strides:", y_copy.strides)
print("x.shape:", x.shape)
print("y.shape:", y.shape)
print("y_copy.shape:", y_copy.shape)


# Benchmark function
def benchmark(name, func, repeat=10*7):
    times = []
    for _ in range(repeat):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    avg = sum(times) / repeat
    print(f"{name:<25}: {avg * 1e3:.3f} ms")
    return avg * 1e3  # return in ms


# Measure execution time
results = {}
results["x.sum()"] = benchmark("x.sum()", lambda: x.sum())
results["y.sum() (non-contig)"] = benchmark("y.sum() (non-contig)", lambda: y.sum())
results["y_copy.sum() (contig)"] = benchmark(
    "y_copy.sum() (contig)", lambda: y_copy.sum()
)

# ====== Visualization ======

labels = list(results.keys())
values = list(results.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=["steelblue", "tomato", "seagreen"])
plt.ylabel("Avg execution time [ms]")
plt.title("NumPy: Effect of Stride and Memory Layout on Performance")
plt.grid(axis="y")
plt.tight_layout()

# Ensure directory exists
os.makedirs("chapter01", exist_ok=True)
plt.savefig("chapter01/stride_benchmark.png")
plt.show()
