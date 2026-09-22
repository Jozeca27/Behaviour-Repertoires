#!/usr/bin/env python3

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

DISTANCE_ROOT = Path(
    "SaveResults/CombinedvsDistance/distance"
)

COMBINED_ROOT = Path(
    "SaveResults/CombinedvsDistance/combined"
)

OUTPUT_DIR = Path(
    "SaveResults/CombinedvsDistance/analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE_EVOLUTION = (10.0, 7.5)
FIGSIZE_FINAL = (10.0, 7.5)

DPI = 300


# ==========================================================
# PLOT STYLE
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": DPI,
})


# ==========================================================
# LOAD ONE CHECKPOINT
# ==========================================================

def load_statistics(checkpoint):

    with open(checkpoint, "rb") as f:
        archive = pickle.load(f)

    if isinstance(archive, dict):

        grid = archive["grid"]

        # --------------------------------------------------
        # Empty archive protection
        # --------------------------------------------------

        if len(grid) == 0:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        fitnesses = np.array([
            c["fitness"]
            for c in grid.values()
        ])

        distances = np.array([
            c["distance"]
            for c in grid.values()
        ])

        speeds = np.array([
            c["average_speed"]
            for c in grid.values()
        ])

        energies = np.array([
            c["energy"]
            for c in grid.values()
        ])

        stabilities = np.array([
            c["stability_penalty"]
            for c in grid.values()
        ])

        coverage = (
            len(grid) /
            np.prod(archive["grid_shape"])
        )

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
# LOAD ALL RUNS FROM ONE EXPERIMENT
# ==========================================================

def load_all_runs(root_dir):

    runs = []

    if not root_dir.exists():
        print(f"WARNING: Folder does not exist: {root_dir}")
        return runs

    for run in sorted(root_dir.iterdir()):

        if not run.is_dir():
            continue

        checkpoint_dir = run / "checkpoints"

        if not checkpoint_dir.exists():
            continue

        checkpoints = sorted(
            checkpoint_dir.glob("map_elites_iter_*.pkl"),
            key=lambda p: int(
                p.stem.split("_")[-1]
            )
        )

        if not checkpoints:
            continue

        iterations = []

        coverage = []
        qd = []
        maximum = []

        distance = []
        speed = []
        energy = []
        stability = []

        for ckpt in checkpoints:

            iteration = int(
                ckpt.stem.split("_")[-1]
            )

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

    return runs


# ==========================================================
# LOAD EXPERIMENTS
# ==========================================================

print("=" * 70)
print("LOADING EXPERIMENTS")
print("=" * 70)

distance_runs = load_all_runs(DISTANCE_ROOT)
combined_runs = load_all_runs(COMBINED_ROOT)

print(
    f"Distance fitness runs: {len(distance_runs)}"
)

print(
    f"Combined fitness runs: {len(combined_runs)}"
)

if len(distance_runs) == 0:
    raise RuntimeError(
        "No distance-fitness runs were found."
    )

if len(combined_runs) == 0:
    raise RuntimeError(
        "No combined-fitness runs were found."
    )


# ==========================================================
# HELPER: STACK RUNS
# ==========================================================

def stack_metric(runs, metric):

    return np.vstack([
        run[metric]
        for run in runs
    ])


# ==========================================================
# FIGURE 1
#
# EVOLUTION OF ARCHIVE COMPONENTS
#
# Distance vs Combined fitness
# Mean +/- 1 SD
# ==========================================================

def plot_evolution_comparison(
    distance_runs,
    combined_runs,
):

    metrics = [
        (
            "distance",
            "Mean archive distance",
        ),
        (
            "speed",
            "Mean archive speed",
        ),
        (
            "energy",
            "Mean archive energy",
        ),
        (
            "stability",
            "Mean archive stability penalty",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE_EVOLUTION,
        sharex=False,
    )

    axes = axes.flatten()

    for ax, (metric, ylabel) in zip(
        axes,
        metrics,
    ):

        # --------------------------------------------------
        # Distance fitness
        # --------------------------------------------------

        distance_values = stack_metric(
            distance_runs,
            metric,
        )

        distance_iterations = (
            distance_runs[0]["iterations"]
        )

        distance_mean = np.nanmean(
            distance_values,
            axis=0,
        )

        distance_std = np.nanstd(
            distance_values,
            axis=0,
        )

        ax.plot(
            distance_iterations,
            distance_mean,
            linewidth=2.5,
            label="Distance fitness",
        )

        ax.fill_between(
            distance_iterations,
            distance_mean - distance_std,
            distance_mean + distance_std,
            alpha=0.20,
        )

        # --------------------------------------------------
        # Combined fitness
        # --------------------------------------------------

        combined_values = stack_metric(
            combined_runs,
            metric,
        )

        combined_iterations = (
            combined_runs[0]["iterations"]
        )

        combined_mean = np.nanmean(
            combined_values,
            axis=0,
        )

        combined_std = np.nanstd(
            combined_values,
            axis=0,
        )

        ax.plot(
            combined_iterations,
            combined_mean,
            linewidth=2.5,
            label="Combined fitness",
        )

        ax.fill_between(
            combined_iterations,
            combined_mean - combined_std,
            combined_mean + combined_std,
            alpha=0.20,
        )

        # --------------------------------------------------
        # Formatting
        # --------------------------------------------------

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------
    # Common legend
    # ------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_pdf = (
        OUTPUT_DIR /
        "component_evolution_distance_vs_combined.pdf"
    )

    output_png = (
        OUTPUT_DIR /
        "component_evolution_distance_vs_combined.png"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_pdf}"
    )

    print(
        f"Saved: {output_png}"
    )


# ==========================================================
# FIGURE 2
#
# FINAL ARCHIVE COMPONENT COMPARISON
#
# Distance vs Combined fitness
# Mean +/- SD at final iteration
# ==========================================================

def plot_final_comparison(
    distance_runs,
    combined_runs,
):

    metrics = [
        (
            "distance",
            "Distance",
        ),
        (
            "speed",
            "Average speed",
        ),
        (
            "energy",
            "Energy",
        ),
        (
            "stability",
            "Stability penalty",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE_FINAL,
    )

    axes = axes.flatten()

    for ax, (metric, title) in zip(
        axes,
        metrics,
    ):

        # --------------------------------------------------
        # Get final values from each run
        # --------------------------------------------------

        distance_final = np.array([
            run[metric][-1]
            for run in distance_runs
        ])

        combined_final = np.array([
            run[metric][-1]
            for run in combined_runs
        ])

        # --------------------------------------------------
        # Means
        # --------------------------------------------------

        distance_mean = np.nanmean(
            distance_final
        )

        combined_mean = np.nanmean(
            combined_final
        )

        # --------------------------------------------------
        # Standard deviations
        # --------------------------------------------------

        distance_std = np.nanstd(
            distance_final
        )

        combined_std = np.nanstd(
            combined_final
        )

        # --------------------------------------------------
        # Bar plot
        # --------------------------------------------------

        x = np.array([0, 1])

        means = np.array([
            distance_mean,
            combined_mean,
        ])

        stds = np.array([
            distance_std,
            combined_std,
        ])

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            width=0.6,
        )

        # --------------------------------------------------
        # Formatting
        # --------------------------------------------------

        ax.set_title(title)

        ax.set_xticks(x)

        ax.set_xticklabels([
            "Distance",
            "Combined",
        ])

        ax.set_ylabel("Final archive mean")

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --------------------------------------------------
        # Add values above bars
        # --------------------------------------------------

        for bar, mean, std in zip(
            bars,
            means,
            stds,
        ):

            if np.isfinite(mean):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    mean * 0.03,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    fig.suptitle(
        "Final archive component comparison",
        fontsize=16,
        y=1.02,
    )

    fig.tight_layout()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_pdf = (
        OUTPUT_DIR /
        "final_component_comparison_distance_vs_combined.pdf"
    )

    output_png = (
        OUTPUT_DIR /
        "final_component_comparison_distance_vs_combined.png"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_pdf}"
    )

    print(
        f"Saved: {output_png}"
    )


# ==========================================================
# RUN PLOTS
# ==========================================================

print()
print("=" * 70)
print("CREATING FIGURE 1")
print("=" * 70)

plot_evolution_comparison(
    distance_runs,
    combined_runs,
)


print()
print("=" * 70)
print("CREATING FIGURE 2")
print("=" * 70)

plot_final_comparison(
    distance_runs,
    combined_runs,
)


print()
print("=" * 70)
print("FINISHED")
print("=" * 70)

print(
    f"Output directory: {OUTPUT_DIR}"
)