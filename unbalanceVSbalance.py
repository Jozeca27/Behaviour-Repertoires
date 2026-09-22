#!/usr/bin/env python3

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

BALANCE_ROOT = Path(
    "SaveResults/BalanceVsUnbalance/transformer_latent/transformer"
)

UNBALANCE_ROOT = Path(
    "SaveResults/BalanceVsUnbalance/transformer_latent/trans2"
)

OUTPUT_DIR = Path(
    "SaveResults/BalanceVsUnbalance/transformer_latent/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------
# Behaviour descriptor configuration
# ----------------------------------------------------------
#
# Change these if your xy_position descriptor uses different
# bounds or grid dimensions.
#
# Example:
#
# X: [-3, 3]
# Y: [-3, 3]
#
# 50 x 50 = 2500 cells
# ----------------------------------------------------------

BD_BOUNDS = [
    (-5.0, 5.0),   # X
    (-5.0, 5.0),   # Y
]

BD_LABELS = [
    "Latent 0",
    "Latent 1",
]


# ----------------------------------------------------------
# Plot configuration
# ----------------------------------------------------------

DPI = 300

FIGSIZE_EVOLUTION = (8.0, 5.5)
FIGSIZE_DIRECTIONAL = (8.0, 5.5)
FIGSIZE_HEATMAP = (7.5, 6.0)
FIGSIZE_FINAL = (9.0, 6.5)

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
# GENERAL HELPERS
# ==========================================================

def get_checkpoint_iteration(path):

    return int(
        path.stem.split("_")[-1]
    )


def get_grid_shape(archive):

    if "grid_shape" in archive:
        return tuple(
            int(x)
            for x in archive["grid_shape"]
        )

    raise KeyError(
        "Could not find 'grid_shape' in archive."
    )


def get_occupied_indices(archive):

    grid = archive["grid"]

    indices = []

    for key in grid.keys():

        # --------------------------------------------------
        # Normal case:
        # key = (x_index, y_index)
        # --------------------------------------------------

        if isinstance(key, tuple):
            idx = tuple(
                int(x)
                for x in key
            )

        # --------------------------------------------------
        # Some implementations may use lists
        # --------------------------------------------------

        elif isinstance(key, list):
            idx = tuple(
                int(x)
                for x in key
            )

        # --------------------------------------------------
        # NumPy array key
        # --------------------------------------------------

        elif isinstance(key, np.ndarray):
            idx = tuple(
                int(x)
                for x in key.tolist()
            )

        else:
            raise ValueError(
                f"Unsupported grid key type: "
                f"{type(key)} | key={key}"
            )

        indices.append(idx)

    return np.asarray(
        indices,
        dtype=int,
    )


# ==========================================================
# LOAD ONE CHECKPOINT
# ==========================================================

def load_archive(checkpoint):

    with open(checkpoint, "rb") as f:
        archive = pickle.load(f)

    if not isinstance(archive, dict):

        raise TypeError(
            f"Expected dictionary archive in "
            f"{checkpoint}, got {type(archive)}"
        )

    if "grid" not in archive:

        raise KeyError(
            f"'grid' not found in {checkpoint}"
        )

    return archive


# ==========================================================
# CONVERT GRID INDEX TO DESCRIPTOR VALUE
# ==========================================================

def cell_index_to_value(
    index,
    dimension,
    grid_shape,
):

    minimum, maximum = BD_BOUNDS[dimension]

    n_cells = grid_shape[dimension]

    # Cell centre
    value = (
        minimum
        + (index + 0.5)
        * (maximum - minimum)
        / n_cells
    )

    return value


# ==========================================================
# CALCULATE DESCRIPTOR METRICS
# ==========================================================

def calculate_descriptor_metrics(archive):

    grid_shape = get_grid_shape(archive)

    occupied = get_occupied_indices(archive)

    total_cells = np.prod(grid_shape)

    occupied_cells = len(occupied)

    coverage = (
        occupied_cells /
        total_cells
    )

    # ------------------------------------------------------
    # Empty archive
    # ------------------------------------------------------

    if occupied_cells == 0:

        return {
            "coverage": 0.0,

            "x_min": np.nan,
            "x_max": np.nan,
            "y_min": np.nan,
            "y_max": np.nan,

            "x_range": 0.0,
            "y_range": 0.0,

            "x_negative": 0.0,
            "x_positive": 0.0,
            "y_negative": 0.0,
            "y_positive": 0.0,

            "x_negative_pct": 0.0,
            "x_positive_pct": 0.0,
            "y_negative_pct": 0.0,
            "y_positive_pct": 0.0,
        }

    # ------------------------------------------------------
    # Convert cell indices to actual descriptor values
    # ------------------------------------------------------

    x_values = np.array([
        cell_index_to_value(
            idx[0],
            0,
            grid_shape,
        )
        for idx in occupied
    ])

    y_values = np.array([
        cell_index_to_value(
            idx[1],
            1,
            grid_shape,
        )
        for idx in occupied
    ])

    x_min = np.min(x_values)
    x_max = np.max(x_values)

    y_min = np.min(y_values)
    y_max = np.max(y_values)

    # ------------------------------------------------------
    # Total physical range explored
    # ------------------------------------------------------

    x_range = x_max - x_min
    y_range = y_max - y_min

    # ------------------------------------------------------
    # Directional expansion
    #
    # How far did the archive reach from zero?
    # ------------------------------------------------------

    x_negative = max(
        0.0,
        -x_min,
    )

    x_positive = max(
        0.0,
        x_max,
    )

    y_negative = max(
        0.0,
        -y_min,
    )

    y_positive = max(
        0.0,
        y_max,
    )

    # ------------------------------------------------------
    # Maximum available expansion in each direction
    # ------------------------------------------------------

    x_min_bound = BD_BOUNDS[0][0]
    x_max_bound = BD_BOUNDS[0][1]

    y_min_bound = BD_BOUNDS[1][0]
    y_max_bound = BD_BOUNDS[1][1]

    x_negative_max = abs(x_min_bound)
    x_positive_max = abs(x_max_bound)

    y_negative_max = abs(y_min_bound)
    y_positive_max = abs(y_max_bound)

    # ------------------------------------------------------
    # Normalized directional coverage
    # ------------------------------------------------------

    x_negative_pct = (
        x_negative / x_negative_max
        if x_negative_max > 0
        else np.nan
    )

    x_positive_pct = (
        x_positive / x_positive_max
        if x_positive_max > 0
        else np.nan
    )

    y_negative_pct = (
        y_negative / y_negative_max
        if y_negative_max > 0
        else np.nan
    )

    y_positive_pct = (
        y_positive / y_positive_max
        if y_positive_max > 0
        else np.nan
    )

    return {
        "coverage": coverage,

        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,

        "x_range": x_range,
        "y_range": y_range,

        "x_negative": x_negative,
        "x_positive": x_positive,
        "y_negative": y_negative,
        "y_positive": y_positive,

        "x_negative_pct": x_negative_pct,
        "x_positive_pct": x_positive_pct,
        "y_negative_pct": y_negative_pct,
        "y_positive_pct": y_positive_pct,
    }


# ==========================================================
# LOAD ALL RUNS
# ==========================================================

def load_all_runs(root_dir):

    runs = []

    if not root_dir.exists():

        print(
            f"WARNING: Directory does not exist: "
            f"{root_dir}"
        )

        return runs

    for run_dir in sorted(root_dir.iterdir()):

        if not run_dir.is_dir():
            continue

        checkpoint_dir = (
            run_dir / "checkpoints"
        )

        if not checkpoint_dir.exists():
            continue

        checkpoints = sorted(
            checkpoint_dir.glob(
                "map_elites_iter_*.pkl"
            ),
            key=get_checkpoint_iteration,
        )

        if not checkpoints:
            continue

        iterations = []

        coverage = []

        x_range = []
        y_range = []

        x_negative = []
        x_positive = []

        y_negative = []
        y_positive = []

        x_negative_pct = []
        x_positive_pct = []

        y_negative_pct = []
        y_positive_pct = []

        final_archive = None

        for checkpoint in checkpoints:

            iteration = get_checkpoint_iteration(
                checkpoint
            )

            archive = load_archive(
                checkpoint
            )

            metrics = (
                calculate_descriptor_metrics(
                    archive
                )
            )

            iterations.append(iteration)

            coverage.append(
                metrics["coverage"]
            )

            x_range.append(
                metrics["x_range"]
            )

            y_range.append(
                metrics["y_range"]
            )

            x_negative.append(
                metrics["x_negative"]
            )

            x_positive.append(
                metrics["x_positive"]
            )

            y_negative.append(
                metrics["y_negative"]
            )

            y_positive.append(
                metrics["y_positive"]
            )

            x_negative_pct.append(
                metrics["x_negative_pct"]
            )

            x_positive_pct.append(
                metrics["x_positive_pct"]
            )

            y_negative_pct.append(
                metrics["y_negative_pct"]
            )

            y_positive_pct.append(
                metrics["y_positive_pct"]
            )

            final_archive = archive

        runs.append({
            "iterations": np.asarray(
                iterations
            ),

            "coverage": np.asarray(
                coverage
            ),

            "x_range": np.asarray(
                x_range
            ),

            "y_range": np.asarray(
                y_range
            ),

            "x_negative": np.asarray(
                x_negative
            ),

            "x_positive": np.asarray(
                x_positive
            ),

            "y_negative": np.asarray(
                y_negative
            ),

            "y_positive": np.asarray(
                y_positive
            ),

            "x_negative_pct": np.asarray(
                x_negative_pct
            ),

            "x_positive_pct": np.asarray(
                x_positive_pct
            ),

            "y_negative_pct": np.asarray(
                y_negative_pct
            ),

            "y_positive_pct": np.asarray(
                y_positive_pct
            ),

            "final_archive": final_archive,
        })

    return runs


# ==========================================================
# LOAD BALANCE / UNBALANCE
# ==========================================================

print("=" * 70)
print("LOADING BALANCE / UNBALANCE EXPERIMENTS")
print("=" * 70)

balance_runs = load_all_runs(
    BALANCE_ROOT
)

unbalance_runs = load_all_runs(
    UNBALANCE_ROOT
)

print(
    f"Balance runs:   {len(balance_runs)}"
)

print(
    f"Unbalance runs: {len(unbalance_runs)}"
)

if len(balance_runs) == 0:
    raise RuntimeError(
        "No Balance runs were found."
    )

if len(unbalance_runs) == 0:
    raise RuntimeError(
        "No Unbalance runs were found."
    )


# ==========================================================
# STACK METRICS
# ==========================================================

def stack_metric(
    runs,
    metric,
):

    return np.vstack([
        run[metric]
        for run in runs
    ])


# ==========================================================
# GENERIC EVOLUTION PLOT
# ==========================================================

def plot_two_experiments(
    balance_runs,
    unbalance_runs,
    metric,
    ylabel,
    filename,
    percentage=False,
):

    balance_values = stack_metric(
        balance_runs,
        metric,
    )

    unbalance_values = stack_metric(
        unbalance_runs,
        metric,
    )

    balance_iterations = (
        balance_runs[0]["iterations"]
    )

    unbalance_iterations = (
        unbalance_runs[0]["iterations"]
    )

    balance_mean = np.nanmean(
        balance_values,
        axis=0,
    )

    balance_std = np.nanstd(
        balance_values,
        axis=0,
    )

    unbalance_mean = np.nanmean(
        unbalance_values,
        axis=0,
    )

    unbalance_std = np.nanstd(
        unbalance_values,
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_EVOLUTION
    )

    # ------------------------------------------------------
    # Balance
    # ------------------------------------------------------

    ax.plot(
        balance_iterations,
        balance_mean,
        linewidth=2.5,
        label="Balance",
    )

    ax.fill_between(
        balance_iterations,
        balance_mean - balance_std,
        balance_mean + balance_std,
        alpha=0.20,
    )

    # ------------------------------------------------------
    # Unbalance
    # ------------------------------------------------------

    ax.plot(
        unbalance_iterations,
        unbalance_mean,
        linewidth=2.5,
        label="Unbalance",
    )

    ax.fill_between(
        unbalance_iterations,
        unbalance_mean - unbalance_std,
        unbalance_mean + unbalance_std,
        alpha=0.20,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)

    if percentage:
        ax.set_ylim(
            0,
            1.05,
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==========================================================
# FIGURE 1
#
# COVERAGE
# ==========================================================

plot_two_experiments(
    balance_runs,
    unbalance_runs,
    "coverage",
    "Archive coverage",
    "coverage_balance_vs_unbalance",
)


# ==========================================================
# FIGURE 2
#
# PER-DIMENSION EXPANSION
# ==========================================================

def plot_dimension_expansion(
    balance_runs,
    unbalance_runs,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    dimensions = [
        ("x_range", "X position range"),
        ("y_range", "Y position range"),
    ]

    for ax, (metric, ylabel) in zip(
        axes,
        dimensions,
    ):

        balance_values = stack_metric(
            balance_runs,
            metric,
        )

        unbalance_values = stack_metric(
            unbalance_runs,
            metric,
        )

        balance_iterations = (
            balance_runs[0]["iterations"]
        )

        unbalance_iterations = (
            unbalance_runs[0]["iterations"]
        )

        balance_mean = np.nanmean(
            balance_values,
            axis=0,
        )

        balance_std = np.nanstd(
            balance_values,
            axis=0,
        )

        unbalance_mean = np.nanmean(
            unbalance_values,
            axis=0,
        )

        unbalance_std = np.nanstd(
            unbalance_values,
            axis=0,
        )

        # Balance
        ax.plot(
            balance_iterations,
            balance_mean,
            linewidth=2.5,
            label="Balance",
        )

        ax.fill_between(
            balance_iterations,
            balance_mean - balance_std,
            balance_mean + balance_std,
            alpha=0.20,
        )

        # Unbalance
        ax.plot(
            unbalance_iterations,
            unbalance_mean,
            linewidth=2.5,
            label="Unbalance",
        )

        ax.fill_between(
            unbalance_iterations,
            unbalance_mean - unbalance_std,
            unbalance_mean + unbalance_std,
            alpha=0.20,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "per_dimension_expansion.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR /
        "per_dimension_expansion.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_dimension_expansion(
    balance_runs,
    unbalance_runs,
)


# ==========================================================
# FIGURE 3
#
# DIRECTIONAL COVERAGE
# ==========================================================

def plot_directional_coverage(
    balance_runs,
    unbalance_runs,
):

    metrics = [
        (
            "x_negative_pct",
            "X negative",
        ),
        (
            "x_positive_pct",
            "X positive",
        ),
        (
            "y_negative_pct",
            "Y negative",
        ),
        (
            "y_positive_pct",
            "Y positive",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7.5),
        sharey=True,
    )

    axes = axes.flatten()

    for ax, (metric, title) in zip(
        axes,
        metrics,
    ):

        balance_values = stack_metric(
            balance_runs,
            metric,
        )

        unbalance_values = stack_metric(
            unbalance_runs,
            metric,
        )

        balance_iterations = (
            balance_runs[0]["iterations"]
        )

        unbalance_iterations = (
            unbalance_runs[0]["iterations"]
        )

        balance_mean = np.nanmean(
            balance_values,
            axis=0,
        )

        balance_std = np.nanstd(
            balance_values,
            axis=0,
        )

        unbalance_mean = np.nanmean(
            unbalance_values,
            axis=0,
        )

        unbalance_std = np.nanstd(
            unbalance_values,
            axis=0,
        )

        # Balance
        ax.plot(
            balance_iterations,
            balance_mean,
            linewidth=2.5,
            label="Balance",
        )

        ax.fill_between(
            balance_iterations,
            balance_mean - balance_std,
            balance_mean + balance_std,
            alpha=0.20,
        )

        # Unbalance
        ax.plot(
            unbalance_iterations,
            unbalance_mean,
            linewidth=2.5,
            label="Unbalance",
        )

        ax.fill_between(
            unbalance_iterations,
            unbalance_mean - unbalance_std,
            unbalance_mean + unbalance_std,
            alpha=0.20,
        )

        ax.set_title(title)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Directional coverage")

        ax.set_ylim(
            0,
            1.05,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

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

    fig.savefig(
        OUTPUT_DIR /
        "directional_coverage.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR /
        "directional_coverage.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_directional_coverage(
    balance_runs,
    unbalance_runs,
)


# ==========================================================
# HEATMAP HELPERS
# ==========================================================

def create_occupancy_matrix(
    archive
):

    grid_shape = get_grid_shape(
        archive
    )

    occupancy = np.zeros(
        grid_shape,
        dtype=float,
    )

    occupied = get_occupied_indices(
        archive
    )

    for idx in occupied:

        if (
            0 <= idx[0] < grid_shape[0]
            and
            0 <= idx[1] < grid_shape[1]
        ):

            occupancy[
                idx[0],
                idx[1]
            ] = 1.0

    return occupancy


def create_mean_occupancy_matrix(
    runs
):

    matrices = []

    for run in runs:

        archive = run["final_archive"]

        matrices.append(
            create_occupancy_matrix(
                archive
            )
        )

    return np.mean(
        np.stack(matrices),
        axis=0,
    )


# ==========================================================
# HEATMAP: ONE EXPERIMENT
# ==========================================================

def plot_heatmap(
    occupancy,
    title,
    filename,
):

    grid_shape = occupancy.shape

    x_min, x_max = BD_BOUNDS[0]
    y_min, y_max = BD_BOUNDS[1]

    fig, ax = plt.subplots(
        figsize=FIGSIZE_HEATMAP
    )

    # ------------------------------------------------------
    # Transpose because matrix rows correspond to Y
    # ------------------------------------------------------

    image = ax.imshow(
        occupancy.T,
        origin="lower",
        extent=[
            x_min,
            x_max,
            y_min,
            y_max,
        ],
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel(
        BD_LABELS[0]
    )

    ax.set_ylabel(
        BD_LABELS[1]
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Fraction of runs occupied"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR /
        f"{filename}.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==========================================================
# FIGURE 4A
#
# BALANCE HEATMAP
# ==========================================================

balance_occupancy = (
    create_mean_occupancy_matrix(
        balance_runs
    )
)

plot_heatmap(
    balance_occupancy,
    "Balance: final descriptor-space occupancy",
    "heatmap_balance",
)


# ==========================================================
# FIGURE 4B
#
# UNBALANCE HEATMAP
# ==========================================================

unbalance_occupancy = (
    create_mean_occupancy_matrix(
        unbalance_runs
    )
)

plot_heatmap(
    unbalance_occupancy,
    "Unbalance: final descriptor-space occupancy",
    "heatmap_unbalance",
)


# ==========================================================
# FIGURE 4C
#
# DIFFERENCE HEATMAP
# ==========================================================

def plot_difference_heatmap(
    balance_occupancy,
    unbalance_occupancy,
):

    difference = (
        balance_occupancy
        -
        unbalance_occupancy
    )

    x_min, x_max = BD_BOUNDS[0]
    y_min, y_max = BD_BOUNDS[1]

    max_abs = np.max(
        np.abs(difference)
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_HEATMAP
    )

    image = ax.imshow(
        difference.T,
        origin="lower",
        extent=[
            x_min,
            x_max,
            y_min,
            y_max,
        ],
        aspect="equal",
        interpolation="nearest",
        vmin=-max_abs,
        vmax=max_abs,
        cmap="RdBu_r",
    )

    ax.set_xlabel(
        BD_LABELS[0]
    )

    ax.set_ylabel(
        BD_LABELS[1]
    )

    ax.set_title(
        "Descriptor-space occupancy difference\n"
        "Balance − Unbalance"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        "Occupancy difference"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "heatmap_balance_minus_unbalance.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR /
        "heatmap_balance_minus_unbalance.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


plot_difference_heatmap(
    balance_occupancy,
    unbalance_occupancy,
)


# ==========================================================
# FINAL NUMERICAL SUMMARY
# ==========================================================

def print_final_summary(
    name,
    runs,
):

    metrics = [
        "coverage",
        "x_range",
        "y_range",
        "x_negative_pct",
        "x_positive_pct",
        "y_negative_pct",
        "y_positive_pct",
    ]

    print()
    print("=" * 70)
    print(f"{name.upper()} FINAL RESULTS")
    print("=" * 70)

    for metric in metrics:

        values = np.array([
            run[metric][-1]
            for run in runs
        ])

        mean = np.nanmean(values)
        std = np.nanstd(values)

        if "pct" in metric:

            print(
                f"{metric:20s}: "
                f"{mean * 100:7.2f}% "
                f"+/- {std * 100:6.2f}%"
            )

        else:

            print(
                f"{metric:20s}: "
                f"{mean:10.4f} "
                f"+/- {std:10.4f}"
            )


print_final_summary(
    "Balance",
    balance_runs,
)

print_final_summary(
    "Unbalance",
    unbalance_runs,
)


# ==========================================================
# FINISHED
# ==========================================================

print()
print("=" * 70)
print("ANALYSIS FINISHED")
print("=" * 70)

print(
    f"Results saved to:\n{OUTPUT_DIR}"
)