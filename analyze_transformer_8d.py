#!/usr/bin/env python3

"""
Batch analysis of MAP-Elites final archives with an 8-D Transformer latent BD.

Expected directory structure:

SaveResults/
└── BalanceVsUnbalance/
    └── xy_position/
        ├── Combined/
        │   ├── balance/
        │   │   ├── run1/
        │   │   │   └── map_elites_final.pkl
        │   │   └── ...
        │   └── unbalance/
        │       └── ...
        │
        └── Distance/
            ├── balance/
            │   └── ...
            └── unbalance/
                └── ...

The script:

1. Finds every map_elites_final.pkl.
2. Reconstructs every archived controller.
3. Calculates:
   - archive coverage
   - spatial uniformity
   - occupied cells
   - fitness correlation
   - local fitness smoothness
   - trajectory diversity
   - cluster silhouette
   - PCA variance
   - descriptor-space expansion
   - directional descriptor coverage
4. Saves one report per run.
5. Saves a CSV containing all experiments.
6. Produces comparison plots.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

import config

from simulation.simulation_final import (
    HexapodTorqueSimulation,
)

from utils.analysis import (
    extract_trajectory_features,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT_DIR = Path(
    "SaveResults/BalanceVsUnbalance/transformer_8d/2"
)

OUTPUT_DIR = (
    ROOT_DIR / "batch_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------
# Descriptor bounds
# ----------------------------------------------------------

BD_DIM = 8

# Bounds of the 8-dimensional Transformer latent behavioural descriptor.
# All eight latent dimensions use the same [-5, 5] range.
BD_BOUNDS = [(-10.0, 10.0)] * BD_DIM



# ----------------------------------------------------------
# Analysis configuration
# ----------------------------------------------------------

N_CLUSTERS = 6
K_NEIGHBORS = 10
SEED = 42

# ==========================================================
# ANALYSIS STAGES
# ==========================================================

RUN_STAGE_1 = True
RUN_STAGE_2 = True

# Number of controllers reconstructed per archive
# for Stage 2.
MAX_CELLS = 200

# Whether to save the individual trajectory pickle files.
SAVE_TRAJECTORIES = False


# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

EXPERIMENT_NAME = "Transformer_Latent_Unbalance"

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seeds(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ==========================================================
# ARCHIVE
# ==========================================================

def load_archive(
    archive_path: Path,
):

    with open(
        archive_path,
        "rb",
    ) as f:

        archive = pickle.load(f)

    if (
        not isinstance(archive, dict)
        or "grid" not in archive
    ):

        raise ValueError(
            f"Unsupported archive format: "
            f"{archive_path}"
        )

    return archive


# ==========================================================
# CONFIGURATION SYNC
# ==========================================================

def sync_config_from_archive(
    archive,
):

    bd_mode = archive.get(
        "bd_mode",
        config.BD_MODE,
    )

    config.BD_MODE = bd_mode

    config.TRANSFORMER_BD_CONFIG = (
        config.get_transformer_bd_config(
            bd_mode
        )
        if bd_mode
        in config.TRANSFORMER_BD_CONFIGS
        else None
    )


# ==========================================================
# GRID HELPERS
# ==========================================================

def get_grid_shape(
    archive,
):

    return tuple(
        int(x)
        for x in archive["grid_shape"]
    )


def get_occupied_indices(
    archive,
):

    indices = []

    for key in archive["grid"].keys():

        if isinstance(key, tuple):

            idx = tuple(
                int(x)
                for x in key
            )

        elif isinstance(key, list):

            idx = tuple(
                int(x)
                for x in key
            )

        elif isinstance(key, np.ndarray):

            idx = tuple(
                int(x)
                for x in key.tolist()
            )

        else:

            raise ValueError(
                f"Unsupported grid key: "
                f"{key}"
            )

        indices.append(idx)

    return np.asarray(
        indices,
        dtype=int,
    )


# ==========================================================
# CELL INDEX -> BD VALUE
# ==========================================================

def cell_index_to_value(
    index,
    dimension,
    grid_shape,
):

    minimum, maximum = (
        BD_BOUNDS[dimension]
    )

    n_cells = grid_shape[dimension]

    return (
        minimum
        +
        (index + 0.5)
        *
        (maximum - minimum)
        /
        n_cells
    )


# ==========================================================
# DESCRIPTOR SPACE METRICS
# ==========================================================

def compute_descriptor_metrics(
    archive,
):
    """
    Compute descriptor-space metrics for an arbitrary-dimensional BD.

    For an 8-D BD, metrics are calculated independently for every latent
    dimension.  There is no attempt to reduce the archive to X/Y or to use
    2-D quadrants.

    Reported metrics:
      - archive_coverage: occupied cells / total grid cells
      - occupied_cells
      - per-dimension min/max/range
      - per-dimension range fraction
      - per-dimension negative/positive reach
      - per-dimension negative/positive reach fraction
      - per-dimension directional balance
      - per-dimension occupied-bin fraction
      - mean/std of the per-dimension metrics
      - marginal spatial uniformity, averaged across dimensions
    """

    grid_shape = get_grid_shape(archive)
    occupied = get_occupied_indices(archive)

    bd_dim = len(grid_shape)

    if len(BD_BOUNDS) != bd_dim:
        raise ValueError(
            f"BD_BOUNDS has {len(BD_BOUNDS)} dimensions, "
            f"but archive grid has {bd_dim} dimensions."
        )

    total_cells = int(np.prod(grid_shape))
    occupied_cells = len(occupied)

    coverage = (
        occupied_cells / total_cells
        if total_cells > 0
        else 0.0
    )

    if occupied_cells == 0:
        metrics = {
            "archive_coverage": 0.0,
            "occupied_cells": 0,
            "total_cells": total_cells,
            "spatial_uniformity": 0.0,
            "range_fraction_mean": 0.0,
            "range_fraction_std": 0.0,
            "directional_balance_mean": 0.0,
            "directional_balance_std": 0.0,
            "occupied_bin_fraction_mean": 0.0,
            "occupied_bin_fraction_std": 0.0,
        }

        for d in range(bd_dim):
            metrics.update({
                f"latent_{d}_min": np.nan,
                f"latent_{d}_max": np.nan,
                f"latent_{d}_range": 0.0,
                f"latent_{d}_range_fraction": 0.0,
                f"latent_{d}_negative_reach": 0.0,
                f"latent_{d}_positive_reach": 0.0,
                f"latent_{d}_negative_fraction": 0.0,
                f"latent_{d}_positive_fraction": 0.0,
                f"latent_{d}_directional_balance": 0.0,
                f"latent_{d}_occupied_bin_fraction": 0.0,
                f"latent_{d}_marginal_uniformity": 0.0,
            })

        return metrics

    # Convert occupied cell indices to the centre of each BD cell.
    bd_values = np.column_stack([
        np.array([
            cell_index_to_value(idx, d, grid_shape)
            for idx in occupied
        ])
        for d in range(bd_dim)
    ])

    metrics = {
        "archive_coverage": coverage,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
    }

    range_fractions = []
    directional_balances = []
    occupied_bin_fractions = []
    marginal_uniformities = []

    for d in range(bd_dim):
        minimum, maximum = BD_BOUNDS[d]
        values = bd_values[:, d]

        d_min = float(np.min(values))
        d_max = float(np.max(values))
        d_range = d_max - d_min

        total_range = maximum - minimum
        range_fraction = (
            d_range / total_range
            if total_range > 0
            else 0.0
        )

        negative_reach = max(0.0, -d_min)
        positive_reach = max(0.0, d_max)

        negative_max = abs(minimum)
        positive_max = abs(maximum)

        negative_fraction = (
            negative_reach / negative_max
            if negative_max > 0
            else 0.0
        )
        positive_fraction = (
            positive_reach / positive_max
            if positive_max > 0
            else 0.0
        )

        directional_balance = (
            min(negative_fraction, positive_fraction)
            / max(negative_fraction, positive_fraction)
            if max(negative_fraction, positive_fraction) > 0
            else 0.0
        )

        # How much of the available bins along this individual dimension
        # contain at least one occupied archive cell.
        unique_bins = len(np.unique(occupied[:, d]))
        occupied_bin_fraction = (
            unique_bins / grid_shape[d]
            if grid_shape[d] > 0
            else 0.0
        )

        # Marginal uniformity: compare the occupied-bin distribution with a
        # perfectly uniform distribution over all bins in this dimension.
        bin_counts = np.bincount(
            occupied[:, d],
            minlength=grid_shape[d],
        ).astype(float)

        if occupied_cells > 0:
            probabilities = bin_counts / occupied_cells
            target = 1.0 / grid_shape[d]
            tv_distance = 0.5 * np.sum(
                np.abs(probabilities - target)
            )
            marginal_uniformity = 1.0 - tv_distance
        else:
            marginal_uniformity = 0.0

        metrics.update({
            f"latent_{d}_min": d_min,
            f"latent_{d}_max": d_max,
            f"latent_{d}_range": d_range,
            f"latent_{d}_range_fraction": range_fraction,
            f"latent_{d}_negative_reach": negative_reach,
            f"latent_{d}_positive_reach": positive_reach,
            f"latent_{d}_negative_fraction": negative_fraction,
            f"latent_{d}_positive_fraction": positive_fraction,
            f"latent_{d}_directional_balance": directional_balance,
            f"latent_{d}_occupied_bin_fraction": occupied_bin_fraction,
            f"latent_{d}_marginal_uniformity": marginal_uniformity,
        })

        range_fractions.append(range_fraction)
        directional_balances.append(directional_balance)
        occupied_bin_fractions.append(occupied_bin_fraction)
        marginal_uniformities.append(marginal_uniformity)

    metrics["range_fraction_mean"] = float(np.mean(range_fractions))
    metrics["range_fraction_std"] = float(np.std(range_fractions))
    metrics["directional_balance_mean"] = float(np.mean(directional_balances))
    metrics["directional_balance_std"] = float(np.std(directional_balances))
    metrics["occupied_bin_fraction_mean"] = float(np.mean(occupied_bin_fractions))
    metrics["occupied_bin_fraction_std"] = float(np.std(occupied_bin_fractions))

    # This replaces the old 2-D quadrant metric. It is the mean marginal
    # uniformity across all latent dimensions.
    metrics["spatial_uniformity"] = float(np.mean(marginal_uniformities))

    return metrics


# ==========================================================
# ORIGINAL ARCHIVE METRICS
# ==========================================================

def compute_fitness_correlation(
    bd,
    fitness,
    k=10,
):

    if len(bd) <= k:

        k = len(bd) - 1

    if k < 1:

        return float("nan")

    nbrs = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(bd)

    _, idx = nbrs.kneighbors(
        bd
    )

    bd_diffs = []
    fit_diffs = []

    for i in range(len(bd)):

        neighbors = idx[i][1:]

        bd_diffs.extend(
            np.linalg.norm(
                bd[i]
                -
                bd[neighbors],
                axis=1,
            )
        )

        fit_diffs.extend(
            np.abs(
                fitness[i]
                -
                fitness[neighbors]
            )
        )

    if len(bd_diffs) < 2:

        return float("nan")

    return float(
        np.corrcoef(
            bd_diffs,
            fit_diffs,
        )[0, 1]
    )


def compute_local_fitness_smoothness(
    bd,
    fitness,
    k=10,
):

    if len(bd) <= k:

        k = len(bd) - 1

    if k < 1:

        return float("nan")

    nbrs = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(bd)

    _, idx = nbrs.kneighbors(
        bd
    )

    neighbor_fitness = (
        fitness[idx[:, 1:]]
    )

    diffs = np.abs(
        fitness[:, None]
        -
        neighbor_fitness
    )

    return float(
        diffs.mean()
    )


def compute_trajectory_diversity(
    features,
):

    if len(features) < 2:

        return 0.0

    d = pairwise_distances(
        features,
        metric="cosine",
    )

    return float(
        d.mean()
    )


def compute_cluster_separation(
    features,
    n_clusters=6,
):

    if (
        len(features)
        <
        n_clusters
    ):

        return (
            float("nan"),
            np.zeros(
                len(features),
                dtype=np.int64,
            ),
        )

    labels = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=42,
    ).fit_predict(
        features
    )

    # Silhouette requires at least
    # 2 distinct clusters.
    if len(
        np.unique(labels)
    ) < 2:

        return (
            float("nan"),
            labels,
        )

    score = silhouette_score(
        features,
        labels,
    )

    return (
        float(score),
        labels,
    )


def compute_pca_variance(
    features_scaled,
    max_components=10,
):

    n_components = min(
        max_components,
        features_scaled.shape[0],
        features_scaled.shape[1],
    )

    if n_components < 1:

        return []

    pca = PCA(
        n_components=n_components
    )

    pca.fit(
        features_scaled
    )

    return (
        pca.explained_variance_ratio_
        .tolist()
    )


# ==========================================================
# TRAJECTORY RECONSTRUCTION
# ==========================================================

@dataclass
class ReconstructedTrajectory:

    grid_idx: Tuple[int, ...]

    archive_fitness: float

    archive_bd: Optional[np.ndarray]

    genome: np.ndarray

    result: Dict[str, Any]


def evaluate_controller_no_goal(
    sim,
    genome,
):

    result = sim.evaluate_controller(
        genome,
        duration=(
            config.SIMULATION_CONFIG[
                "duration"
            ]
        ),
        warmup_time=(
            config.SIMULATION_CONFIG[
                "warmup_time"
            ]
        ),
        sample_interval=(
            config.SIMULATION_CONFIG[
                "sample_interval"
            ]
        ),
        controller_interval=(
            config.SIMULATION_CONFIG.get(
                "controller_interval",
                4,
            )
        ),
        log_all_steps=True,
    )

    return result


def reconstruct_archive(
    archive,
    output_dir,
    save_trajectories=True,
    max_cells=None,
):

    grid = archive["grid"]

    if not grid:

        raise ValueError(
            "Archive grid is empty."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_dir = (
        output_dir
        /
        "reconstructed_trajectories"
    )

    if save_trajectories:

        trajectory_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ------------------------------------------------------
    # Highest fitness first
    # ------------------------------------------------------

    items = sorted(
        grid.items(),
        key=lambda item:
            item[1]["fitness"],
        reverse=True,
    )

    if max_cells is not None:

        items = items[
            :max_cells
        ]


    print(
        f"Reconstructing "
        f"{len(items)} archive cells..."
    )


    sim = HexapodTorqueSimulation(
        urdf_path=config.URDF_PATH,
        gui=False,
        time_step=(
            config.SIMULATION_CONFIG[
                "time_step"
            ]
        ),
    )


    records = []

    try:

        for rank, (
            grid_idx,
            cell,
        ) in enumerate(
            items,
            start=1,
        ):

            if rank % 100 == 0:

                print(
                    f"  Reconstructed "
                    f"{rank}/{len(items)}"
                )


            result = (
                evaluate_controller_no_goal(
                    sim,
                    cell["genome"],
                )
            )


            record = (
                ReconstructedTrajectory(
                    grid_idx=grid_idx,
                    archive_fitness=float(
                        cell["fitness"]
                    ),
                    archive_bd=(
                        np.asarray(
                            cell.get("bd"),
                            dtype=np.float32,
                        )
                        if cell.get("bd")
                        is not None
                        else None
                    ),
                    genome=np.asarray(
                        cell["genome"],
                        dtype=np.float32,
                    ),
                    result=result,
                )
            )

            records.append(
                record
            )


            if save_trajectories:

                file_name = (
                    f"cell_{rank:05d}_"
                    f"idx_{'_'.join(map(str, grid_idx))}.pkl"
                )

                with open(
                    trajectory_dir / file_name,
                    "wb",
                ) as f:

                    pickle.dump(
                        {
                            "grid_idx": grid_idx,
                            "archive_fitness": record.archive_fitness,
                            "archive_bd": record.archive_bd,
                            "genome": record.genome,
                            "result": result,
                        },
                        f,
                    )

    finally:

        sim.disconnect()


    return records


# ==========================================================
# BUILD REPORT
# ==========================================================

def build_stage2_report(
    archive,
    records,
    n_clusters,
):

    # ------------------------------------------------------
    # Reconstructed BD
    # ------------------------------------------------------

    bd = np.array([
        np.asarray(
            record.result[
                "behavior_descriptors"
            ],
            dtype=np.float32,
        )
        for record in records
    ])


    # ------------------------------------------------------
    # Fitness
    # ------------------------------------------------------

    fitness = np.array([
        float(
            record.result.get(
                "fitness",
                record.archive_fitness,
            )
        )
        for record in records
    ])


    # ------------------------------------------------------
    # Trajectory features
    # ------------------------------------------------------

    features = np.array([
        extract_trajectory_features(
            record.result.get(
                "detailed_log",
                [],
            )
        )
        for record in records
    ])


    # ------------------------------------------------------
    # Remove problematic rows
    # ------------------------------------------------------

    valid = (
        np.all(
            np.isfinite(features),
            axis=1,
        )
    )

    features = features[
        valid
    ]

    bd = bd[
        valid
    ]

    fitness = fitness[
        valid
    ]


    if len(features) == 0:

        raise RuntimeError(
            "No valid reconstructed "
            "trajectory features."
        )


    # ------------------------------------------------------
    # Standardize
    # ------------------------------------------------------

    scaler = StandardScaler()

    features_scaled = (
        scaler.fit_transform(
            features
        )
    )


    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    explained = (
        compute_pca_variance(
            features_scaled
        )
    )


    # ------------------------------------------------------
    # Clustering
    # ------------------------------------------------------

    cluster_score, labels = (
        compute_cluster_separation(
            features_scaled,
            n_clusters=n_clusters,
        )
    )


    # ------------------------------------------------------
    # Local metrics
    # ------------------------------------------------------

    if len(records) >= 2:

        smooth_k = min(
            K_NEIGHBORS,
            len(records) - 1,
        )

        smoothness = (
            compute_local_fitness_smoothness(
                bd,
                fitness,
                k=smooth_k,
            )
        )

        fit_corr = (
            compute_fitness_correlation(
                bd,
                fitness,
                k=smooth_k,
            )
        )

        traj_div = (
            compute_trajectory_diversity(
                features_scaled
            )
        )

    else:

        smoothness = float("nan")

        fit_corr = float("nan")

        traj_div = 0.0


    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    report = {

        "total_cells":
            int(
                np.prod(
                    archive[
                        "grid_shape"
                    ]
                )
            ),

        "fitness_correlation":
            fit_corr,

        "local_fitness_smoothness":
            smoothness,

        "trajectory_diversity":
            traj_div,

        "cluster_silhouette":
            cluster_score,

        "pca_explained_variance":
            explained,

        "pca_2d_total":
            (
                float(
                    sum(
                        explained[:2]
                    )
                )
                if explained
                else 0.0
            ),

        "pca_8d_total":
            (
                float(
                    sum(
                        explained[:BD_DIM]
                    )
                )
                if explained
                else 0.0
            ),

        "archive_cells_replayed":
            len(records),

        "valid_trajectory_features":
            len(features),
    }


    return (
        report,
        bd,
        fitness,
        labels,
    )


def build_stage1_report(archive):

    descriptor_metrics = (
        compute_descriptor_metrics(
            archive
        )
    )

    descriptor_metrics["total_cells"] = int(
        np.prod(
            archive["grid_shape"]
        )
    )

    return descriptor_metrics



# ==========================================================
# SAVE ONE RUN
# ==========================================================

def save_run_outputs(
    output_dir,
    report,
    bd,
    fitness,
    labels,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_dir / "report.json",
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            allow_nan=True,
        )


    with open(
        output_dir
        /
        "reconstructed_summary.pkl",
        "wb",
    ) as f:

        pickle.dump(
            {
                "bd":
                    bd,

                "fitness":
                    fitness,

                "labels":
                    labels,
            },
            f,
        )


# ==========================================================
# FIND FINAL ARCHIVES
# ==========================================================

def find_final_archives(
    root_dir,
):

    archives = sorted(
        root_dir.rglob(
            "map_elites_final.pkl"
        )
    )

    return archives


# ==========================================================
# FLATTEN REPORT FOR CSV
# ==========================================================

def flatten_report(
    condition,
    run_name,
    archive_path,
    report,
):

    row = {

        "condition":
            condition,

        "run":
            run_name,

        "archive":
            str(archive_path),

    }


    # ------------------------------------------------------
    # Scalar metrics
    # ------------------------------------------------------

    for key, value in report.items():

        if isinstance(
            value,
            (list, tuple),
        ):

            continue

        if isinstance(
            value,
            np.generic,
        ):

            value = value.item()

        row[key] = value


    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    pca = report.get(
        "pca_explained_variance",
        [],
    )

    for i, value in enumerate(
        pca
    ):

        row[
            f"pca_{i + 1}_variance"
        ] = value


    return row


# ==========================================================
# SAVE CSV
# ==========================================================

def save_csv(
    rows,
    output_path,
):

    if not rows:

        return

    # Get union of all fields
    fields = []

    for row in rows:

        for key in row.keys():

            if key not in fields:

                fields.append(
                    key
                )


    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

def create_summary(
    rows,
):

    conditions = sorted(
        set(
            row["condition"]
            for row in rows
        )
    )

    summary_rows = []

    metric_names = [
        "archive_coverage",
        "spatial_uniformity",
        "range_fraction_mean",
        "directional_balance_mean",
        "occupied_bin_fraction_mean",
        "fitness_correlation",
        "local_fitness_smoothness",
        "trajectory_diversity",
        "cluster_silhouette",
        "pca_2d_total",
        "pca_8d_total",
    ]


    for condition in conditions:

        condition_rows = [
            row
            for row in rows
            if row["condition"]
            == condition
        ]


        summary = {
            "condition":
                condition,

            "n_runs":
                len(
                    condition_rows
                ),
        }


        for metric in metric_names:

            values = []

            for row in condition_rows:

                value = row.get(
                    metric
                )

                if value is None:

                    continue

                try:

                    value = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if np.isfinite(
                    value
                ):

                    values.append(
                        value
                    )


            if values:

                summary[
                    f"{metric}_mean"
                ] = np.mean(
                    values
                )

                summary[
                    f"{metric}_std"
                ] = np.std(
                    values
                )

            else:

                summary[
                    f"{metric}_mean"
                ] = np.nan

                summary[
                    f"{metric}_std"
                ] = np.nan


        summary_rows.append(
            summary
        )


    return summary_rows


# ==========================================================
# PLOT FINAL METRICS
# ==========================================================

def plot_metric_comparison(
    rows,
    metric,
    ylabel,
    filename,
    percentage=False,
):
    """
    Plot a scalar metric for the available experimental conditions.

    The original script hard-coded four 2-D XY conditions.  This version
    uses the conditions actually present in the analysed rows, so it also
    works for a single 8-D Transformer experiment.
    """

    conditions = sorted(
        set(row["condition"] for row in rows)
    )

    data = []
    labels = []

    for condition in conditions:
        values = [
            float(row[metric])
            for row in rows
            if row["condition"] == condition
            and row.get(metric) is not None
            and np.isfinite(float(row[metric]))
        ]

        if values:
            data.append(values)
            labels.append(
                condition.replace("_", "\n")
            )

    if not data:
        return

    means = np.array([
        np.mean(values)
        for values in data
    ])

    stds = np.array([
        np.std(values)
        for values in data
    ])

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(len(data))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        width=0.65,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)

    if percentage:
        ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mean in zip(bars, means):
        if np.isfinite(mean):
            text = (
                f"{mean * 100:.1f}%"
                if percentage
                else f"{mean:.3f}"
            )

            y = mean * 0.03 if mean >= 0 else mean * 0.03

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_per_dimension_metric(
    rows,
    metric_suffix,
    ylabel,
    filename,
    percentage=False,
):
    """
    Plot an 8-D metric with one bar per latent dimension.

    If multiple runs/conditions exist, the values are averaged per
    condition first and then displayed side-by-side for each dimension.
    """

    conditions = sorted(
        set(row["condition"] for row in rows)
    )

    dimensions = range(BD_DIM)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = np.arange(BD_DIM)

    valid_conditions = []

    for condition in conditions:
        means = []
        stds = []

        for d in dimensions:
            key = f"latent_{d}_{metric_suffix}"

            values = [
                float(row[key])
                for row in rows
                if row["condition"] == condition
                and row.get(key) is not None
                and np.isfinite(float(row[key]))
            ]

            means.append(
                np.mean(values) if values else np.nan
            )
            stds.append(
                np.std(values) if values else 0.0
            )

        if np.any(np.isfinite(means)):
            valid_conditions.append(
                (condition, np.asarray(means), np.asarray(stds))
            )

    n_conditions = max(1, len(valid_conditions))
    width = 0.8 / n_conditions

    for i, (condition, means, stds) in enumerate(valid_conditions):
        offset = (
            i - (n_conditions - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=condition.replace("_", "\n"),
        )

        for bar, mean in zip(bars, means):
            if np.isfinite(mean):
                text = (
                    f"{mean * 100:.1f}%"
                    if percentage
                    else f"{mean:.2f}"
                )

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean * 0.03 if mean >= 0 else mean * 0.03,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Latent {d}" for d in dimensions],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel(ylabel)

    if percentage:
        ax.set_ylim(0, 1.05)

    if valid_conditions:
        ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_comparison_plots(rows):
    # Overall archive metrics.
    plot_metric_comparison(
        rows,
        "archive_coverage",
        "Archive coverage",
        "comparison_coverage",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "spatial_uniformity",
        "Mean marginal uniformity",
        "comparison_spatial_uniformity",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "range_fraction_mean",
        "Mean latent range",
        "comparison_range_fraction",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "directional_balance_mean",
        "Mean directional balance",
        "comparison_directional_balance",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "occupied_bin_fraction_mean",
        "Mean occupied-bin fraction",
        "comparison_occupied_bins",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "fitness_correlation",
        "Fitness correlation",
        "comparison_fitness_correlation",
    )

    plot_metric_comparison(
        rows,
        "local_fitness_smoothness",
        "Local fitness smoothness",
        "comparison_fitness_smoothness",
    )

    plot_metric_comparison(
        rows,
        "trajectory_diversity",
        "Trajectory diversity",
        "comparison_trajectory_diversity",
    )

    plot_metric_comparison(
        rows,
        "cluster_silhouette",
        "Cluster silhouette",
        "comparison_cluster_silhouette",
    )

    plot_metric_comparison(
        rows,
        "pca_2d_total",
        "Variance explained by first 2 PCs",
        "comparison_pca_2d",
        percentage=True,
    )

    plot_metric_comparison(
        rows,
        "pca_8d_total",
        "Variance explained by first 8 PCs",
        "comparison_pca_8d",
        percentage=True,
    )

    # Dimension-specific 8-D plots.
    plot_per_dimension_metric(
        rows,
        "range_fraction",
        "Latent dimension range",
        "comparison_range_per_dimension",
        percentage=True,
    )

    plot_per_dimension_metric(
        rows,
        "directional_balance",
        "Directional balance",
        "comparison_balance_per_dimension",
        percentage=True,
    )

    plot_per_dimension_metric(
        rows,
        "occupied_bin_fraction",
        "Occupied-bin fraction",
        "comparison_occupied_bins_per_dimension",
        percentage=True,
    )

    plot_per_dimension_metric(
        rows,
        "marginal_uniformity",
        "Marginal uniformity",
        "comparison_uniformity_per_dimension",
        percentage=True,
    )


# ==========================================================
# PROCESS ONE EXPERIMENT
# ==========================================================

def process_condition(
    condition,
    root_dir,
    all_stage1_rows,
    all_stage2_rows,
):

    print()
    print("=" * 70)
    print(f"CONDITION: {condition}")
    print("=" * 70)

    archives = find_final_archives(root_dir)

    print(
        f"Found {len(archives)} final archives."
    )

    if not archives:
        print(
            f"WARNING: no archives found in {root_dir}"
        )
        return

    for run_number, archive_path in enumerate(
        archives,
        start=1,
    ):

        run_name = archive_path.parent.parent.name

        if run_name == "checkpoints":
            run_name = (
                archive_path
                .parent
                .parent
                .parent
                .name
            )

        print()
        print("-" * 70)
        print(
            f"{condition} | "
            f"Run {run_number}/{len(archives)}"
        )
        print(f"Archive: {archive_path}")

        archive = load_archive(
            archive_path
        )

        sync_config_from_archive(
            archive
        )

        # ==================================================
        # STAGE 1
        # ==================================================

        if RUN_STAGE_1:

            print()
            print("  [STAGE 1] Archive metrics")

            report = build_stage1_report(
                archive
            )

            report["archive_path"] = os.path.abspath(
                archive_path
            )

            report["condition"] = condition
            report["run"] = run_name
            report["bd_mode"] = archive.get(
                "bd_mode"
            )
            report["fitness_mode"] = archive.get(
                "fitness_mode"
            )

            row = flatten_report(
                condition,
                run_name,
                archive_path,
                report,
            )

            all_stage1_rows.append(
                row
            )

            print(
                f"    Coverage: "
                f"{report['archive_coverage']:.4f}"
            )

            print(
                f"    Mean latent range: "
                f"{report['range_fraction_mean'] * 100:.2f}%"
            )

            print(
                f"    Mean directional balance: "
                f"{report['directional_balance_mean']:.4f}"
            )

            print(
                f"    Mean occupied-bin fraction: "
                f"{report['occupied_bin_fraction_mean'] * 100:.2f}%"
            )

            for d in range(BD_DIM):
                print(
                    f"    Latent {d}: "
                    f"range={report[f'latent_{d}_range_fraction'] * 100:.2f}%, "
                    f"balance={report[f'latent_{d}_directional_balance']:.4f}, "
                    f"bins={report[f'latent_{d}_occupied_bin_fraction'] * 100:.2f}%"
                )

        # ==================================================
        # STAGE 2
        # ==================================================

        if RUN_STAGE_2:

            print()
            print(
                f"  [STAGE 2] Reconstructing "
                f"up to {MAX_CELLS} trajectories"
            )

            run_output = (
                OUTPUT_DIR
                /
                "stage2"
                /
                condition
                /
                run_name
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            records = reconstruct_archive(
                archive=archive,
                output_dir=run_output,
                save_trajectories=SAVE_TRAJECTORIES,
                max_cells=MAX_CELLS,
            )

            (
                report,
                bd,
                fitness,
                labels,
            ) = build_stage2_report(
                archive=archive,
                records=records,
                n_clusters=N_CLUSTERS,
            )

            report["archive_path"] = os.path.abspath(
                archive_path
            )

            report["condition"] = condition
            report["run"] = run_name
            report["bd_mode"] = archive.get(
                "bd_mode"
            )
            report["fitness_mode"] = archive.get(
                "fitness_mode"
            )

            report["archive_cells_replayed"] = (
                len(records)
            )

            save_run_outputs(
                run_output,
                report,
                bd,
                fitness,
                labels,
            )

            row = flatten_report(
                condition,
                run_name,
                archive_path,
                report,
            )

            all_stage2_rows.append(
                row
            )

            print(
                f"    Trajectory diversity: "
                f"{report['trajectory_diversity']:.4f}"
            )

            print(
                f"    Cluster silhouette: "
                f"{report['cluster_silhouette']:.4f}"
            )

            print(
                f"    PCA first 2: "
                f"{report['pca_2d_total'] * 100:.2f}%"
            )

            print(
                f"    PCA first {BD_DIM}: "
                f"{report['pca_8d_total'] * 100:.2f}%"
            )
# ==========================================================
# MAIN
# ==========================================================

def main():

    set_seeds(SEED)

    print()
    print("=" * 70)
    print("BATCH MAP-ELITES ARCHIVE ANALYSIS")
    print("=" * 70)

    print(
        f"Root directory: {ROOT_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()
    print(
        f"Stage 1 enabled: {RUN_STAGE_1}"
    )

    print(
        f"Stage 2 enabled: {RUN_STAGE_2}"
    )

    if RUN_STAGE_2:
        print(
            f"Stage 2 max cells/archive: "
            f"{MAX_CELLS}"
        )

    all_stage1_rows = []
    all_stage2_rows = []

    # ======================================================
    # PROCESS ALL FOUR CONDITIONS
    # ======================================================


    process_condition(
        EXPERIMENT_NAME,
        ROOT_DIR,
        all_stage1_rows,
        all_stage2_rows,
    )

    # ======================================================
    # STAGE 1 OUTPUT
    # ======================================================

    if RUN_STAGE_1:

        stage1_dir = (
            OUTPUT_DIR / "stage1"
        )

        stage1_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stage1_csv = (
            stage1_dir
            /
            "all_experiments_stage1.csv"
        )

        save_csv(
            all_stage1_rows,
            stage1_csv,
        )

        stage1_summary = create_summary(
            all_stage1_rows
        )

        stage1_summary_csv = (
            stage1_dir
            /
            "experiment_summary_stage1.csv"
        )

        save_csv(
            stage1_summary,
            stage1_summary_csv,
        )

        create_comparison_plots(
            all_stage1_rows
        )

    # ======================================================
    # STAGE 2 OUTPUT
    # ======================================================

    if RUN_STAGE_2:

        stage2_dir = (
            OUTPUT_DIR / "stage2"
        )

        stage2_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stage2_csv = (
            stage2_dir
            /
            "all_experiments_stage2.csv"
        )

        save_csv(
            all_stage2_rows,
            stage2_csv,
        )

        stage2_summary = create_summary(
            all_stage2_rows
        )

        stage2_summary_csv = (
            stage2_dir
            /
            "experiment_summary_stage2.csv"
        )

        save_csv(
            stage2_summary,
            stage2_summary_csv,
        )

    # ======================================================
    # FINISHED
    # ======================================================

    print()
    print("=" * 70)
    print("BATCH ANALYSIS FINISHED")
    print("=" * 70)

    print(
        f"Stage 1 archives analysed: "
        f"{len(all_stage1_rows)}"
    )

    print(
        f"Stage 2 archives analysed: "
        f"{len(all_stage2_rows)}"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}"
    )

if __name__ == "__main__":

    main()