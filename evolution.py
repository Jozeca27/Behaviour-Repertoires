#!/usr/bin/env python3

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

YES = False


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT_DIR = Path("SaveResults/BalanceVsUnbalance/transformer_latent/trans2")          # Folder containing all distance runs
#ROOT_DIR = Path("SaveResults/xy_position/BalanceVsUnbalance/Combined")          # Folder containing all combined runs
OUTPUT_DIR = ROOT_DIR / "analysis"

OUTPUT_DIR.mkdir(exist_ok=True)

FIGSIZE = (7.0, 4.5)
DPI = 300

if YES:

    checkpoint = ROOT_DIR / "checkpoints" / "map_elites_iter_0.pkl"

    with open(checkpoint, "rb") as f:
        archive = pickle.load(f)

    print(archive.keys())

    cell = next(iter(archive["grid"].values()))

    print("Distance:", cell["distance"])
    print("Average speed:", cell["average_speed"])
    print("Energy:", cell["energy"])
    print("Stability penalty:", cell["stability_penalty"])
    print(cell.keys())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": DPI
})


# ==========================================================
# LOAD ONE CHECKPOINT
# ==========================================================

def load_statistics(checkpoint):

    with open(checkpoint, "rb") as f:
        archive = pickle.load(f)

    if isinstance(archive, dict):

        grid = archive["grid"]

        fitnesses = np.array([c["fitness"] for c in grid.values()])
        distances = np.array([c["distance"] for c in grid.values()])
        speeds = np.array([c["average_speed"] for c in grid.values()])
        energies = np.array([c["energy"] for c in grid.values()])
        stabilities = np.array([c["stability_penalty"] for c in grid.values()])

        coverage = len(grid) / np.prod(archive["grid_shape"])
        qd_score = np.sum(fitnesses)
        max_fitness = np.max(fitnesses)

        mean_distance = np.mean(distances)
        mean_speed = np.mean(speeds)
        mean_energy = np.mean(energies)
        mean_stability = np.mean(stabilities)

    else:

        stats = archive.get_statistics()

        coverage = stats["coverage"]
        qd_score = stats["qd_score"]
        max_fitness = stats["max_fitness"]

        mean_distance = np.nan
        mean_speed = np.nan
        mean_energy = np.nan
        mean_stability = np.nan

    return (
        coverage,
        qd_score,
        max_fitness,
        mean_distance,
        mean_speed,
        mean_energy,
        mean_stability,
    )   

# ==========================================================
# LOAD ALL RUNS
# ==========================================================

runs = []

for run in sorted(ROOT_DIR.iterdir()):

    checkpoint_dir = run / "checkpoints"

    if not checkpoint_dir.exists():
        continue

    checkpoints = sorted(
        checkpoint_dir.glob("map_elites_iter_*.pkl"),
        key=lambda p: int(p.stem.split("_")[-1])
    )

    iterations = []
    coverage = []
    qd = []
    maximum = []
    distance = []
    speed = []
    energy = []
    stability = []

    for ckpt in checkpoints:

        iteration = int(ckpt.stem.split("_")[-1])

        (
            c,
            q,
            m,
            d,
            s,
            e,
            st,
        ) = load_statistics(ckpt)

        iterations.append(iteration)
        coverage.append(c)
        qd.append(q)
        maximum.append(m)
        distance.append(d)
        speed.append(s)
        energy.append(e)
        stability.append(st)

    runs.append({
        "iterations": np.asarray(iterations),
        "coverage": np.asarray(coverage),
        "qd": np.asarray(qd),
        "max": np.asarray(maximum),
        "distance": np.asarray(distance),
        "speed": np.asarray(speed),
        "energy": np.asarray(energy),
        "stability": np.asarray(stability),
    })

print(f"Loaded {len(runs)} runs.")


# ==========================================================
# STACK RESULTS
# ==========================================================

iterations = runs[0]["iterations"]

coverage = np.vstack([r["coverage"] for r in runs])
qd = np.vstack([r["qd"] for r in runs])
maximum = np.vstack([r["max"] for r in runs])
distance = np.vstack([r["distance"] for r in runs])
speed = np.vstack([r["speed"] for r in runs])
energy = np.vstack([r["energy"] for r in runs])
stability = np.vstack([r["stability"] for r in runs])

def plot_metric(values, ylabel, filename):

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    plt.figure(figsize=FIGSIZE)

    # ----------------------------------------------------------
    # Individual runs
    # ----------------------------------------------------------
    for run in values:
        plt.plot(
            iterations,
            run,
            color="0.75",      # light gray
            linewidth=1.0,
            alpha=0.8,
            zorder=1
        )

    # ----------------------------------------------------------
    # Standard deviation
    # ----------------------------------------------------------
    plt.fill_between(
        iterations,
        mean - std,
        mean + std,
        alpha=0.25,
        zorder=2,
        label="±1 SD"
    )

    # ----------------------------------------------------------
    # Mean
    # ----------------------------------------------------------
    plt.plot(
        iterations,
        mean,
        linewidth=3,
        color="C0",
        label="Mean",
        zorder=3
    )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)

    plt.grid(True, alpha=0.3)

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{filename}.png",
                dpi=300,
                bbox_inches="tight")

    plt.close()

plot_metric(coverage, "Coverage", "coverage_vs_iteration")

plot_metric(qd, "QD-score", "qd_score_vs_iteration")

plot_metric(maximum, "Maximum fitness", "max_fitness_vs_iteration")

plot_metric(
    distance,
    "Mean archive distance",
    "distance_vs_iteration",
)

plot_metric(
    speed,
    "Mean archive speed",
    "average_speed_vs_iteration",
)

plot_metric(
    energy,
    "Mean archive energy",
    "energy_vs_iteration",
)

plot_metric(
    stability,
    "Mean archive stability penalty",
    "stability_vs_iteration",
)



print("Finished.")