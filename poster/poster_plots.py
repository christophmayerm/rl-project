"""
Plot generator for the poster.

"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = Path(__file__).resolve().parent

RUNS: Dict[str, Path] = {
    "gp": ROOT / "data/racetrack4_T1/gp/20251215-150302",
    "greedy": ROOT / "data/racetrack4_T1/greedy/20251215-154740",
}

# Cohesive palette across figures
COLORS = {
    "standard": "#6c6c6c",
    "fspmi_greedy": "#1f77b4",
    "fspmi_gp": "#d62728",
    "fsapmi": "#2ca02c",
    "sapmi": "#8c564b",
}


def load_csv(path, delimiter = ","):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_semicolon_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open() as f:
        header = f.readline().lstrip("# ").strip().split(";")
        reader = csv.DictReader(f, fieldnames=header, delimiter=";")
        rows = []
        for row in reader:
            if not row or all(v == "" for v in row.values()):
                continue
            rows.append({k: float(row[k]) for k in header})
    return rows


def to_arrays(rows, x_key, y_key):
    xs, ys = [], []
    for r in rows:
        xs.append(float(r[x_key]))
        ys.append(float(r[y_key]))
    return xs, ys


def downsample(rows, step):
    if step <= 1:
        return list(rows)
    return list(rows[::step])


def first_hit(xs, ys, threshold):
    for x, y in zip(xs, ys):
        if y >= threshold:
            return x
    return None


def cumulative(values):
    total = 0.0
    out = []
    for v in values:
        total += v
        out.append(total)
    return out


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "lines.linewidth": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
        }
    )


def plot_sample_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    thresholds = [0.4, 0.5, 0.6]
    step = 3

    # Left: return vs iterations (greedy chooser)
    mode = "greedy"
    ax = axes[0]
    f_rows = downsample(load_csv(RUNS[mode] / "fspmi_n4.csv"), step)
    s_rows = downsample(load_semicolon_csv(RUNS[mode] / "standard_spmi.csv"), step)
    f_x, f_y = to_arrays(f_rows, "iteration", "performance_true")
    s_x, s_y = to_arrays(s_rows, "iterations", "evaluations")

    ax.plot(s_x, s_y, label="Standard SPMI", color=COLORS["standard"], linestyle="--")
    ax.plot(f_x, f_y, label="F-SPMI (4 agents, greedy)", color=COLORS["fspmi_greedy"])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Return")
    ax.set_title("Greedy chooser: F-SPMI vs Standard")
    ymax = max(max(f_y), max(s_y))
    ax.set_ylim(0.0, ymax + 0.1)

    for t in thresholds:
        f_hit = first_hit(f_x, f_y, t)
        s_hit = first_hit(s_x, s_y, t)
        ax.axhline(t, color="#bbbbbb", linestyle=":", linewidth=1)
        if f_hit is not None:
            ax.axvline(f_hit, color=COLORS["fspmi_greedy"], linestyle=":", linewidth=1.5)
        if s_hit is not None:
            ax.axvline(s_hit, color=COLORS["standard"], linestyle=":", linewidth=1.5)
        text_y = min(ymax, t + 0.08)
        if f_hit is not None and s_hit is not None:
            speedup = s_hit / f_hit
            ax.text(
                0.02 * max(f_x),
                text_y,
                f"{speedup:.1f}× to {t}",
                color=COLORS["fspmi_greedy"],
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        elif s_hit is None:
            ax.text(
                0.02 * max(f_x),
                text_y,
                f"Standard never reaches {t}",
                color=COLORS["standard"],
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
    ax.legend(loc="lower right")

    # Right: MC optimism vs true performance (federated)
    ax = axes[1]
    ax.set_title("MAYBE A PLOT OF THE RACETRACK?")

    fig.suptitle("Sample efficiency & MISSING", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sample_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_safety():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    step = 4

    for ax, mode in zip(axes, ["greedy", "gp"]):
        fspmi_rows = downsample(load_csv(RUNS[mode] / "fspmi_n4.csv"), step)
        fsapmi_rows = downsample(load_csv(RUNS[mode] / "fsapmi_constant_medium_n4.csv"), step)
        sapmi_rows = downsample(
            load_semicolon_csv(RUNS[mode] / "sapmi_policy_constant_medium.csv"), step
        )

        f_x, f_y = to_arrays(fspmi_rows, "iteration", "performance_true")
        f_bound_x, f_bounds = to_arrays(fspmi_rows, "iteration", "bound")
        fs_x, fs_y = to_arrays(fsapmi_rows, "iteration", "performance_true")
        fs_bound_x, fs_bounds = to_arrays(fsapmi_rows, "iteration", "bound")
        s_x, s_y = to_arrays(sapmi_rows, "iterations", "evaluations")
        s_bound_x, s_bounds = to_arrays(sapmi_rows, "iterations", "bound")

        ax.plot(f_x, f_y, label="F-SPMI return", color=COLORS["fspmi_greedy"] if mode == "greedy" else COLORS["fspmi_gp"])
        ax.plot(fs_x, fs_y, label="F-SA-PMI return", color=COLORS["fsapmi"])
        ax.plot(s_x, s_y, label="SA-PMI return", color=COLORS["sapmi"], linestyle="--")

        ax2 = ax.twinx()
        ax2.plot(f_bound_x, f_bounds, color=COLORS["fspmi_greedy"] if mode == "greedy" else COLORS["fspmi_gp"], linestyle=":", linewidth=1.5, label="F-SPMI bound")
        ax2.plot(fs_bound_x, fs_bounds, color=COLORS["fsapmi"], linestyle=":", linewidth=1.5, label="F-SA-PMI bound")
        ax2.plot(s_bound_x, s_bounds, color=COLORS["sapmi"], linestyle=":", linewidth=1.5, label="SA-PMI bound")

        ax.set_title(f"Returns (solid) & safety bounds (dotted) — {mode}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Return")
        ax2.set_ylabel("Safety bound value")

        # Collect legend handles from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence & Safety signals", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "convergence_safety.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mc_vs_true():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    step = 3

    for ax, mode in zip(axes, ["greedy", "gp"]):
        rows = downsample(load_csv(RUNS[mode] / "fspmi_n4.csv"), step)
        iters, perf_true = to_arrays(rows, "iteration", "performance_true")
        perf_mc = [r["performance_mc"] for r in rows]
        gap = [mc - true for mc, true in zip(perf_mc, perf_true)]

        color = COLORS["fspmi_greedy"] if mode == "greedy" else COLORS["fspmi_gp"]
        ax.plot(iters, perf_true, label="True return", color=color)
        ax.plot(iters, perf_mc, label="MC return", color=color, linestyle="--")
        ax.plot(iters, gap, label="MC - true gap", color=color, linestyle=":")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_title(f"{mode.capitalize()} chooser")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Return / gap")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("MC estimator is optimistic vs exact performance", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mc_vs_true.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    plot_sample_efficiency()
    plot_convergence_safety()
    plot_mc_vs_true()
    print("Saved plots to", PLOT_DIR)


if __name__ == "__main__":
    main()
