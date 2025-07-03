import numpy as np
import matplotlib.pyplot as plt

# Input points
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


# Labels
labels_and = np.array([AND(x[0], x[1]) for x in X])
labels_or = np.array([OR(x[0], x[1]) for x in X])
labels_xor = np.array([XOR(x[0], x[1]) for x in X])


# Plotting function
def plot_gate(X, labels, title, ax, boundary_line=None):
    for x, label in zip(X, labels):
        color = "red" if label == 1 else "blue"
        ax.scatter(
            x[0],
            x[1],
            color=color,
            label=(
                str(label)
                if str(label) not in ax.get_legend_handles_labels()[1]
                else ""
            ),
        )

    ax.set_title(title)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True)
    ax.legend()

    # Draw decision boundary if provided
    if boundary_line is not None:
        x_vals = np.linspace(-0.5, 1.5, 100)
        y_vals = boundary_line(x_vals)
        ax.plot(x_vals, y_vals, "k--")  # dashed black line


# Decision boundaries (from perceptron logic)
# AND: x1 + x2 = 1.5
def and_line(x):
    return (1.5 - 0.5 * x) / 0.5


# OR: x1 + x2 = 0.5
def or_line(x):
    return (0.5 - 0.5 * x) / 0.5


# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_gate(X, labels_and, "AND (Linearly Separable)", axes[0], and_line)
plot_gate(X, labels_or, "OR (Linearly Separable)", axes[1], or_line)
plot_gate(X, labels_xor, "XOR (Not Linearly Separable)", axes[2], None)

plt.tight_layout()
plt.show()
