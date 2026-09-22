#!/usr/bin/env python3

"""
Batch analysis of all Balance/Unbalance x Distance/Combined
MAP-Elites final archives.

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
    "SaveResults/BalanceVsUnbalance/transformer_latent/trans2"
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

BD_BOUNDS = [
    (-5.0, 5.0),   # latent 0
    (-5.0, 5.0),   # latent 1
]


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

EXPERIMENT_NAME = "Transformer_Latent_Unalance"

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

    grid_shape = get_grid_shape(
        archive
    )

    occupied = get_occupied_indices(
        archive
    )

    total_cells = int(
        np.prod(grid_shape)
    )

    occupied_cells = len(
        occupied
    )

    coverage = (
        occupied_cells
        /
        total_cells
        if total_cells > 0
        else 0.0
    )

    if occupied_cells == 0:

        return {
            "archive_coverage": 0.0,
            "occupied_cells": 0,

            "x_min": np.nan,
            "x_max": np.nan,
            "y_min": np.nan,
            "y_max": np.nan,

            "x_range": 0.0,
            "y_range": 0.0,

            "x_range_fraction": 0.0,
            "y_range_fraction": 0.0,

            "x_negative_reach": 0.0,
            "x_positive_reach": 0.0,
            "y_negative_reach": 0.0,
            "y_positive_reach": 0.0,

            "x_negative_fraction": 0.0,
            "x_positive_fraction": 0.0,
            "y_negative_fraction": 0.0,
            "y_positive_fraction": 0.0,

            "directional_balance_x": 0.0,
            "directional_balance_y": 0.0,

            "spatial_uniformity": 0.0,
        }


    # ------------------------------------------------------
    # Convert cell indices to actual BD values
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


    x_min = float(
        np.min(x_values)
    )

    x_max = float(
        np.max(x_values)
    )

    y_min = float(
        np.min(y_values)
    )

    y_max = float(
        np.max(y_values)
    )


    # ------------------------------------------------------
    # Range
    # ------------------------------------------------------

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_total_range = (
        BD_BOUNDS[0][1]
        -
        BD_BOUNDS[0][0]
    )

    y_total_range = (
        BD_BOUNDS[1][1]
        -
        BD_BOUNDS[1][0]
    )

    x_range_fraction = (
        x_range / x_total_range
    )

    y_range_fraction = (
        y_range / y_total_range
    )


    # ------------------------------------------------------
    # Directional reach
    # ------------------------------------------------------

    x_negative_reach = max(
        0.0,
        -x_min,
    )

    x_positive_reach = max(
        0.0,
        x_max,
    )

    y_negative_reach = max(
        0.0,
        -y_min,
    )

    y_positive_reach = max(
        0.0,
        y_max,
    )


    # ------------------------------------------------------
    # Maximum possible directional reach
    # ------------------------------------------------------

    x_negative_max = abs(
        BD_BOUNDS[0][0]
    )

    x_positive_max = abs(
        BD_BOUNDS[0][1]
    )

    y_negative_max = abs(
        BD_BOUNDS[1][0]
    )

    y_positive_max = abs(
        BD_BOUNDS[1][1]
    )


    # ------------------------------------------------------
    # Directional fractions
    # ------------------------------------------------------

    x_negative_fraction = (
        x_negative_reach
        /
        x_negative_max
    )

    x_positive_fraction = (
        x_positive_reach
        /
        x_positive_max
    )

    y_negative_fraction = (
        y_negative_reach
        /
        y_negative_max
    )

    y_positive_fraction = (
        y_positive_reach
        /
        y_positive_max
    )


    # ------------------------------------------------------
    # Directional balance
    #
    # 1.0 = perfectly symmetric
    # 0.0 = entirely one-sided
    # ------------------------------------------------------

    directional_balance_x = (
        min(
            x_negative_fraction,
            x_positive_fraction,
        )
        /
        max(
            x_negative_fraction,
            x_positive_fraction,
        )
        if max(
            x_negative_fraction,
            x_positive_fraction,
        ) > 0
        else 0.0
    )

    directional_balance_y = (
        min(
            y_negative_fraction,
            y_positive_fraction,
        )
        /
        max(
            y_negative_fraction,
            y_positive_fraction,
        )
        if max(
            y_negative_fraction,
            y_positive_fraction,
        ) > 0
        else 0.0
    )


    # ------------------------------------------------------
    # Spatial uniformity
    #
    # Divide the grid into quadrants.
    #
    # Perfectly balanced:
    # 25%, 25%, 25%, 25%
    # ------------------------------------------------------

    center_x = grid_shape[0] / 2
    center_y = grid_shape[1] / 2

    quadrant_counts = np.zeros(
        4,
        dtype=float,
    )

    for idx in occupied:

        x = idx[0]
        y = idx[1]

        if (
            x < center_x
            and y < center_y
        ):

            quadrant_counts[0] += 1

        elif (
            x >= center_x
            and y < center_y
        ):

            quadrant_counts[1] += 1

        elif (
            x < center_x
            and y >= center_y
        ):

            quadrant_counts[2] += 1

        else:

            quadrant_counts[3] += 1


    quadrant_prob = (
        quadrant_counts
        /
        occupied_cells
    )

    target = 0.25

    # Total variation distance from
    # perfect quadrant uniformity.
    tv_distance = (
        0.5
        *
        np.sum(
            np.abs(
                quadrant_prob
                -
                target
            )
        )
    )

    spatial_uniformity = (
        1.0
        -
        tv_distance
    )


    return {

        "archive_coverage": coverage,

        "occupied_cells":
            occupied_cells,

        "x_min":
            x_min,

        "x_max":
            x_max,

        "y_min":
            y_min,

        "y_max":
            y_max,

        "x_range":
            x_range,

        "y_range":
            y_range,

        "x_range_fraction":
            x_range_fraction,

        "y_range_fraction":
            y_range_fraction,

        "x_negative_reach":
            x_negative_reach,

        "x_positive_reach":
            x_positive_reach,

        "y_negative_reach":
            y_negative_reach,

        "y_positive_reach":
            y_positive_reach,

        "x_negative_fraction":
            x_negative_fraction,

        "x_positive_fraction":
            x_positive_fraction,

        "y_negative_fraction":
            y_negative_fraction,

        "y_positive_fraction":
            y_positive_fraction,

        "directional_balance_x":
            directional_balance_x,

        "directional_balance_y":
            directional_balance_y,

        "spatial_uniformity":
            spatial_uniformity,
    }


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

        "x_range_fraction",
        "y_range_fraction",

        "x_negative_fraction",
        "x_positive_fraction",

        "y_negative_fraction",
        "y_positive_fraction",

        "directional_balance_x",
        "directional_balance_y",

        "fitness_correlation",

        "local_fitness_smoothness",

        "trajectory_diversity",

        "cluster_silhouette",

        "pca_2d_total",
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

    conditions = [
        "Distance_Unbalance",
        "Distance_Balance",
        "Combined_Unbalance",
        "Combined_Balance",
    ]


    data = []

    labels = []

    for condition in conditions:

        values = [
            float(
                row[metric]
            )
            for row in rows
            if row["condition"]
            == condition
            and row.get(metric)
            is not None
            and np.isfinite(
                float(
                    row[metric]
                )
            )
        ]

        if values:

            data.append(
                values
            )

            labels.append(
                condition.replace(
                    "_",
                    "\n",
                )
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


    x = np.arange(
        len(data)
    )


    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        width=0.65,
    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        ylabel
    )


    if percentage:

        ax.set_ylim(
            0,
            1.05,
        )


    ax.grid(
        axis="y",
        alpha=0.3,
    )


    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # ------------------------------------------------------
    # Value inside bottom of bars
    # ------------------------------------------------------

    for bar, mean in zip(
        bars,
        means,
    ):

        if np.isfinite(
            mean
        ):

            if percentage:

                text = (
                    f"{mean * 100:.1f}%"
                )

            else:

                text = (
                    f"{mean:.3f}"
                )


            y = (
                mean * 0.03
                if mean >= 0
                else mean * 0.03
            )


            ax.text(
                bar.get_x()
                +
                bar.get_width()
                /
                2,
                y,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
            )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR
        /
        f"{filename}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR
        /
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==========================================================
# PLOT ALL COMPARISONS
# ==========================================================

def create_comparison_plots(
    rows,
):

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
        "Spatial uniformity",
        "comparison_spatial_uniformity",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "x_range_fraction",
        "X descriptor range",
        "comparison_x_range",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "y_range_fraction",
        "Y descriptor range",
        "comparison_y_range",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "x_negative_fraction",
        "X negative reach",
        "comparison_x_negative",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "x_positive_fraction",
        "X positive reach",
        "comparison_x_positive",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "y_negative_fraction",
        "Y negative reach",
        "comparison_y_negative",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "y_positive_fraction",
        "Y positive reach",
        "comparison_y_positive",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "directional_balance_x",
        "X directional balance",
        "comparison_directional_balance_x",
        percentage=True,
    )


    plot_metric_comparison(
        rows,
        "directional_balance_y",
        "Y directional balance",
        "comparison_directional_balance_y",
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
                f"    Latent 0 range: "
                f"{report['x_range_fraction'] * 100:.2f}%"
            )

            print(
                f"    Latent 1 range: "
                f"{report['y_range_fraction'] * 100:.2f}%"
            )

            print(
                f"    Latent 0 balance: "
                f"{report['directional_balance_x']:.4f}"
            )

            print(
                f"    Latent 1 balance: "
                f"{report['directional_balance_y']:.4f}"
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