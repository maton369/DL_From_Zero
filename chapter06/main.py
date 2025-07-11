import sys, os
import statistics
import random

sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import shuffle_dataset
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
import optuna

# ===== シード制御（統計的評価では固定しない）=====
USE_FIXED_SEED = False  # True にすると固定される

if USE_FIXED_SEED:
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
else:
    np.random.seed()
    random.seed()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
x_train, t_train = x_train[:500], t_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val, t_val = x_train[:validation_num], t_train[:validation_num]
x_train, t_train = x_train[validation_num:], t_train[validation_num:]


# ===== モデル学習（テスト精度 + 勾配ノルム返却）=====
def train_model(lr, weight_decay, epochs=50):
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100] * 6,
        output_size=10,
        weight_decay_lambda=weight_decay,
    )
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epochs,
        mini_batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": lr},
        verbose=False,
    )
    trainer.train()

    # 勾配ノルム（最終バッチ）
    grads = network.gradient(x_val, t_val)
    grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

    # 最終テスト精度
    test_acc = network.accuracy(x_test, t_test)
    return test_acc, trainer.test_acc_list, trainer.train_acc_list, grad_norm


def summarize_results(name, results):
    accs = [r["test_acc"] for r in results]
    grads = [r["grad_norm"] for r in results]
    print(f"\n=== {name} 統計結果 ===")
    print(f"Test Accuracy: mean = {np.mean(accs):.4f}, std = {np.std(accs):.4f}")
    print(f"Grad Norm:     mean = {np.mean(grads):.4f}, std = {np.std(grads):.4f}")


# ===== グリッドサーチ =====
print("\n=== グリッドサーチ開始 ===")
grid_lrs = np.logspace(-6, -2, 10)
grid_wds = np.logspace(-8, -4, 10)
grid_summary = []
for repeat_idx in range(10):
    grid_results = []
    grid_trial = 1  # 試行番号のカウント
    for lr in grid_lrs:
        for wd in grid_wds:
            test_acc, val_acc, train_acc, grad_norm = train_model(lr, wd)
            print(f"[Grid-{grid_trial}] test acc={test_acc:.4f} | lr={lr:.6f}, wd={wd:.6f}, grad_norm={grad_norm:.4f}")
            grid_results.append(
                {
                    "lr": lr,
                    "wd": wd,
                    "test_acc": test_acc,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "grad_norm": grad_norm,
                    "trial": grid_trial,  # ← 試行番号記録
                }
            )
            grid_trial += 1

    best_grid = max(grid_results, key=lambda r: r["test_acc"])
    grid_summary.append(best_grid)
summarize_results("Grid Search", grid_summary)

# ===== ARS探索 =====
print("\n=== ARS探索開始 ===")


def adaptive_search(start_lr, end_lr, start_wd, end_wd, delta=0.5, max_iter=10):
    best_score = -1
    best_config = None
    best_result = None
    best_trial = -1  # ← 初期化

    lr_left, lr_right = start_lr, end_lr
    wd_left, wd_right = start_wd, end_wd
    for i in range(max_iter):
        if i % 2 == 0:
            lr, wd = lr_left, wd_left
            lr_left *= 1 + delta
            wd_left *= 1 + delta
        else:
            lr, wd = lr_right, wd_right
            lr_right *= 1 - delta
            wd_right *= 1 - delta

        test_acc, val_acc, train_acc, grad_norm = train_model(lr, wd)
        if test_acc > best_score:
            best_score = test_acc
            best_config = (lr, wd)
            best_result = {
                "test_acc": test_acc,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "grad_norm": grad_norm,
                "trial": i + 1,  # ← 試行番号を記録
            }
        print(
            f"[ARS-{i+1}] test acc={test_acc:.4f} | lr={lr:.6f}, wd={wd:.6f}, grad_norm={grad_norm:.4f}"
        )

    return best_config, best_result


ars_summary = []
for repeat_idx in range(10):
    ars_config, ars_result = adaptive_search(
        start_lr=1e-6, end_lr=1e-2, start_wd=1e-8, end_wd=1e-4, delta=0.5, max_iter=100
    )
    ars_summary.append(ars_result)
summarize_results("ARS Search", ars_summary)

# ===== Optuna (TPE) 最適化 =====
print("\n=== Optuna (TPE) 最適化開始 ===")


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    wd = trial.suggest_loguniform("wd", 1e-8, 1e-4)
    test_acc, _, _, _ = train_model(lr, wd)
    print(f"[Optuna-{trial.number + 1}] test acc={test_acc:.4f} | lr={lr:.6f}, wd={wd:.6f}")
    return test_acc  # maximize


optuna_summary = []
for repeat_idx in range(10):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)

    # 最良パラメータで再学習して詳細取得
    best_params = study.best_params
    optuna_test_acc, optuna_val_acc, optuna_train_acc, optuna_grad_norm = train_model(
        best_params["lr"], best_params["wd"]
    )
    best_trial_number = study.best_trial.number + 1  # 0-indexed → 1-indexed
    optuna_summary.append(
        {
            "test_acc": optuna_test_acc,
            "val_acc": optuna_val_acc,
            "train_acc": optuna_train_acc,
            "grad_norm": optuna_grad_norm,
            "trial": best_trial_number,
            "lr": best_params["lr"],
            "wd": best_params["wd"],
        }
    )
summarize_results("Optuna Search", optuna_summary)

# ===== 可視化 =====
plt.figure(figsize=(10, 4))
x = np.arange(len(ars_summary[0]["val_acc"]))

# ARS
plt.plot(
    x,
    ars_summary[0]["val_acc"],
    label=f"ARS (lr={ars_config[0]:.1e}, wd={ars_config[1]:.1e})",
    marker="o",
)
plt.plot(x, ars_summary[0]["train_acc"], linestyle="--", label="ARS train")

plt.plot(
    x,
    optuna_summary[0]["val_acc"],
    label=f"Optuna (lr={optuna_summary[0]['lr']:.1e}, wd={optuna_summary[0]['wd']:.1e})",
    marker="x",
)
plt.plot(x, optuna_summary[0]["train_acc"], linestyle="--", label="Optuna train")

plt.plot(x, grid_summary[0]["val_acc"], label="Grid", marker="^")
plt.plot(x, grid_summary[0]["train_acc"], linestyle="--", label="Grid train")

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Hyperparameter Search using Test Accuracy & Gradient Norm")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True)
plt.show()

# 勾配ノルムとテスト精度の比較
print("\n=== 結果比較 ===")
print(
    f"ARS:    test acc = {ars_summary[0]['test_acc']:.4f}, grad norm = {ars_summary[0]['grad_norm']:.4f}"
)
print(
    f"Grid:   test acc = {grid_summary[0]['test_acc']:.4f}, grad norm = {grid_summary[0]['grad_norm']:.4f}"
)
print(f"Optuna: test acc = {optuna_summary[0]['test_acc']:.4f}, grad norm = {optuna_summary[0]['grad_norm']:.4f}")

print(f"\n=== 到達試行回数比較 ===")
print(f"Grid:   到達試行 = {grid_summary[0]['trial']} / {len(grid_lrs) * len(grid_wds)}")
print(f"ARS:    到達試行 = {ars_summary[0]['trial']} / 100")
print(f"Optuna: 到達試行 = {optuna_summary[0]['trial']} / {len(study.trials)}")
