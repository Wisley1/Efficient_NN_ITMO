import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load(name):
    with (RESULTS / name).open() as handle:
        rows = list(csv.DictReader(handle))
    iteration = [int(row["iteration"]) for row in rows]
    accuracy = [100 * float(row["test_acc"]) for row in rows]
    learning_rate = [float(row["lr"]) for row in rows]
    return iteration, accuracy, learning_rate


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_regimes():
    base_it, base_acc, _ = load("baseline.csv")
    clr_it, clr_acc, _ = load("clr_b1000.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(base_it, base_acc, color="#1f4e79", lw=1.6, label="piecewise, batch 125")
    ax.plot(clr_it, clr_acc, color="#c45911", lw=1.8, label="clr, batch 1000")
    ax.axhline(91.2, color="#1f4e79", ls="--", lw=0.8, alpha=0.8)
    ax.axhline(92.4, color="#c45911", ls="--", lw=0.8, alpha=0.8)
    ax.set_xlim(0, 80_000)
    ax.set_ylim(0, 100)
    ax.set_xlabel("iteration")
    ax.set_ylabel("test accuracy, %")
    ax.legend(loc="lower right", frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_regimes.png", dpi=160)
    plt.close(fig)


def plot_range():
    iteration, accuracy, learning_rate = load("range.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(learning_rate, accuracy, color="#2e7d32", lw=1.8)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("test accuracy, %")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "lr_range_test.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_regimes()
    if (RESULTS / "range.csv").exists():
        plot_range()
