"""
Plot generator for the poster.

Produces three figures (PNG, 300 dpi) in the poster directory:
  - sample_efficiency.png
  - convergence_safety.png
  - gp_vs_greedy.png

Data sources (fixed to the poster run):
  GP runs:      ../data/racetrack4_T1/gp/20251215-103637
  Greedy runs:  ../data/racetrack4_T1/greedy/20251215-114027
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = Path(__file__).resolve().parent

RUNS: Dict[str, Path] = {
    "gp": ROOT / "data/racetrack4_T1/gp/20251215-103637",
    "greedy": ROOT / "data/racetrack4_T1/greedy/20251215-114027",
}

# Cohesive palette across figures
COLORS = {
    "standard": "#6c6c6c",
    "fspmi_greedy": "#1f77b4",
    "fspmi_gp": "#d62728",
    "fsapmi": "#2ca02c",
    "sapmi": "#8c564b",
}


def load_csv(path: Path, delimiter: str = ",") -> List[Dict[str, float]]:
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


def load_semicolon_csv(path: Path) -> List[Dict[str, float]]:
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


def to_arrays(
    rows: Sequence[Dict[str, float]], x_key: str, y_key: str
) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        xs.append(float(r[x_key]))
        ys.append(float(r[y_key]))
    return xs, ys


def downsample(rows: Sequence[Dict[str, float]], step: int) -> List[Dict[str, float]]:
    if step <= 1:
        return list(rows)
    return list(rows[::step])


def first_hit(xs: Iterable[float], ys: Iterable[float], threshold: float) -> float | None:
    for x, y in zip(xs, ys):
        if y >= threshold:
            return x
    return None


def cumulative(values: Sequence[float]) -> List[float]:
    total = 0.0
    out: List[float] = []
    for v in values:
        total += v
        out.append(total)
    return out


def configure_matplotlib() -> None:
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


def plot_sample_efficiency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    thresholds = [0.4, 0.5, 0.6]
    step = 3  # reduce visual clutter

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

    # Right: return vs cumulative samples (federated only, greedy vs GP)
    ax = axes[1]
    f_rows_greedy = downsample(load_csv(RUNS["greedy"] / "fspmi_n4.csv"), step)
    f_rows_gp = downsample(load_csv(RUNS["gp"] / "fspmi_n4.csv"), step)
    g_samples, g_returns = to_arrays(f_rows_greedy, "total_samples", "performance_true")
    gp_samples, gp_returns = to_arrays(f_rows_gp, "total_samples", "performance_true")
    g_cum = cumulative(g_samples)
    gp_cum = cumulative(gp_samples)

    ax.plot(g_cum, g_returns, label="F-SPMI (greedy chooser)", color=COLORS["fspmi_greedy"])
    ax.plot(gp_cum, gp_returns, label="F-SPMI (GP chooser)", color=COLORS["fspmi_gp"])
    ax.set_xlabel("Cumulative samples (federated only)")
    ax.set_title("Sample usage: greedy vs GP chooser")
    ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=1)
    ax.legend(loc="lower right")

    fig.suptitle("Sample efficiency (returns)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sample_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_safety() -> None:
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


def plot_gp_vs_greedy() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    thresholds = [0.4, 0.5, 0.6]
    methods = ["Standard", "F-SPMI (greedy)", "F-SPMI (GP)"]

    # Use greedy run for standard baseline
    std_rows = load_semicolon_csv(RUNS["greedy"] / "standard_spmi.csv")
    g_rows = load_csv(RUNS["greedy"] / "fspmi_n4.csv")
    gp_rows = load_csv(RUNS["gp"] / "fspmi_n4.csv")

    def time_to_thresh(rows: Sequence[Dict[str, float]], x_key: str, y_key: str) -> Dict[float, Optional[float]]:
        xs, ys = to_arrays(rows, x_key, y_key)
        return {t: first_hit(xs, ys, t) for t in thresholds}

    std_hits = time_to_thresh(std_rows, "iterations", "evaluations")
    g_hits = time_to_thresh(g_rows, "iteration", "performance_true")
    gp_hits = time_to_thresh(gp_rows, "iteration", "performance_true")

    data = [
        [std_hits[t] if std_hits[t] is not None else float("nan") for t in thresholds],
        [g_hits[t] if g_hits[t] is not None else float("nan") for t in thresholds],
        [gp_hits[t] if gp_hits[t] is not None else float("nan") for t in thresholds],
    ]

    x = range(len(thresholds))
    width = 0.22
    offsets = [-width, 0, width]
    colors = [COLORS["standard"], COLORS["fspmi_greedy"], COLORS["fspmi_gp"]]

    for idx, (method, vals) in enumerate(zip(methods, data)):
        ax.bar(
            [p + offsets[idx] for p in x],
            vals,
            width=width,
            label=method,
            color=colors[idx],
        )
        for p, v in zip([p + offsets[idx] for p in x], vals):
            label = "n/a" if v != v else f"{v:.0f}"
            ax.text(p, 0 if v != v else v + 10, label, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"return ≥ {t}" for t in thresholds])
    ax.set_ylabel("Iterations to threshold (lower is better)")
    ax.set_title("Time-to-return thresholds: GP vs greedy vs standard")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gp_vs_greedy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    plot_sample_efficiency()
    plot_convergence_safety()
    plot_gp_vs_greedy()
    print("Saved plots to", PLOT_DIR)


if __name__ == "__main__":
    main()
