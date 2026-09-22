"""
Analyze a saved MAP-Elites final archive by reconstructing trajectories from
the stored genomes.

This script:
- loads `map_elites_final.pkl`
- replays each archived genome with the saved experiment configuration
- captures the resulting detailed trajectory log
- runs the existing BD-quality metrics on the reconstructed data
- writes a JSON report and optional per-cell trajectory pickles

Run from the repository root, for example:

    /home/joze/Mestrado/Dissertacao/Pybullet/bulletenv/bin/python -m utils.analyze_final_archive \
        --archive SaveResults/5000Runs/finalTransformer/map_elites_final.pkl \
        --output SaveResults/5000Runs/5000Trans/bd_analysis_from_archive

    NEW paths
    SaveResults/BalanceVsUnbalance/xy_position/Combined/balance
    SaveResults/BalanceVsUnbalance/xy_position/Combined/unbalance
    SaveResults/BalanceVsUnbalance/xy_position/Distance/balance
    SaveResults/BalanceVsUnbalance/xy_position/Distance/unbalance
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

import config
from simulation.simulation_final import HexapodTorqueSimulation
from utils.analysis import (
    extract_trajectory_features,
    plot_bd,
    plot_clusters,
)


@dataclass
class ReconstructedTrajectory:
    grid_idx: Tuple[int, ...]
    archive_fitness: float
    archive_bd: Optional[np.ndarray]
    genome: np.ndarray
    result: Dict[str, Any]


def load_archive(archive_path: str) -> Dict[str, Any]:
    with open(archive_path, "rb") as f:
        archive = pickle.load(f)

    if not isinstance(archive, dict) or "grid" not in archive:
        raise ValueError(f"Unsupported archive format in {archive_path}")

    return archive


def _sync_config_from_archive(archive: Dict[str, Any]) -> None:
    bd_mode = archive.get("bd_mode", config.BD_MODE)
    config.BD_MODE = bd_mode
    config.TRANSFORMER_BD_CONFIG = (
        config.get_transformer_bd_config(bd_mode)
        if bd_mode in config.TRANSFORMER_BD_CONFIGS
        else None
    )


def _set_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluation_settings() -> Dict[str, Any]:
    return {
        "duration": config.SIMULATION_CONFIG["duration"],
        "warmup_time": config.SIMULATION_CONFIG.get("warmup_time", 1.0),
        "sample_interval": config.SIMULATION_CONFIG["sample_interval"],
        "controller_interval": config.SIMULATION_CONFIG.get("controller_interval", 4),
    }


def reconstruct_archive_trajectories(
    archive: Dict[str, Any],
    output_dir: str,
    max_cells: Optional[int] = None,
    save_trajectories: bool = True,
) -> List[ReconstructedTrajectory]:
    grid = archive["grid"]
    if not grid:
        raise ValueError("The archive grid is empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_path / "reconstructed_trajectories"
    if save_trajectories:
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(
        grid.items(),
        key=lambda item: item[1]["fitness"],
        reverse=True,
    )
    if max_cells is not None:
        items = items[:max_cells]

    sim = HexapodTorqueSimulation(
        urdf_path=config.URDF_PATH,
        gui=False,
        time_step=config.SIMULATION_CONFIG["time_step"],
    )

    records: List[ReconstructedTrajectory] = []
    try:
        for rank, (grid_idx, cell) in enumerate(items, start=1):
            result = evaluate_controller_no_goal(
                sim,
                cell["genome"],
            )

            record = ReconstructedTrajectory(
                grid_idx=grid_idx,
                archive_fitness=float(cell["fitness"]),
                archive_bd=np.asarray(cell.get("bd"), dtype=np.float32)
                if cell.get("bd") is not None
                else None,
                genome=np.asarray(cell["genome"], dtype=np.float32),
                result=result,
            )
            records.append(record)

            if save_trajectories:
                file_name = f"cell_{rank:05d}_idx_{'_'.join(map(str, grid_idx))}.pkl"
                with open(trajectories_dir / file_name, "wb") as f:
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

def compute_archive_coverage(
    archive: Dict[str, Any]
) -> float:

    occupied = len(archive["grid"])

    total = int(
        np.prod(
            archive["grid_shape"]
        )
    )

    return occupied / total if total > 0 else 0.0

def compute_archive_uniformity(archive: dict) -> float:
    """
    Uniformity over the MAP-Elites grid.

    We assume each occupied cell has equal weight (elite stored once per cell).
    This measures how evenly the archive fills the grid.
    """

    grid_shape = archive["grid_shape"]
    total_cells = int(np.prod(grid_shape))

    occupied_cells = len(archive["grid"])

    if occupied_cells == 0:
        return 0.0

    # Each filled cell contributes equally
    p = np.ones(occupied_cells, dtype=np.float64)
    p = p / p.sum()

    entropy = -np.sum(p * np.log(p + 1e-12))

    # Normalize by max entropy over FULL grid (important!)
    max_entropy = np.log(total_cells)

    return entropy / max_entropy if max_entropy > 0 else 0.0

def compute_fitness_correlation(bd, fitness, k=10):
    """
    Correlation between behavior distance and fitness difference
    in local neighborhoods.
    """

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(bd)
    _, idx = nbrs.kneighbors(bd)

    bd_diffs = []
    fit_diffs = []

    for i in range(len(bd)):
        neighbors = idx[i][1:]

        bd_diffs.extend(
            np.linalg.norm(bd[i] - bd[neighbors], axis=1)
        )

        fit_diffs.extend(
            np.abs(fitness[i] - fitness[neighbors])
        )

    if len(bd_diffs) < 2:
        return float("nan")

    return float(np.corrcoef(bd_diffs, fit_diffs)[0, 1])

def compute_local_fitness_smoothness(bd, fitness, k=10):
    """
    Average absolute fitness difference in kNN neighborhood.
    """

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(bd)
    _, idx = nbrs.kneighbors(bd)

    neighbor_fitness = fitness[idx[:, 1:]]

    diffs = np.abs(
        fitness[:, None] - neighbor_fitness
    )

    return float(diffs.mean())

def compute_trajectory_diversity(features):
    """
    Mean pairwise cosine distance between trajectories.
    """

    if len(features) < 2:
        return 0.0

    d = pairwise_distances(features, metric="cosine")

    return float(d.mean())

def compute_cluster_separation(features, n_clusters=6):
    """
    Measures how separable trajectory behaviors are.
    """

    if len(features) < n_clusters:
        return float("nan"), np.zeros(len(features))

    labels = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=42
    ).fit_predict(features)

    score = silhouette_score(features, labels)

    return float(score), labels

def compute_pca_variance(features_scaled, max_components=10):
    """
    Fraction of variance explained by PCA.
    """

    n_components = min(
        max_components,
        features_scaled.shape[0],
        features_scaled.shape[1],
    )

    if n_components < 1:
        return []

    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)

    return pca.explained_variance_ratio_.tolist()

def build_report(
    archive: Dict[str, Any],
    records: List[ReconstructedTrajectory],
    n_clusters: int,
):
    bd = np.array([
        np.asarray(
            record.result["behavior_descriptors"],
            dtype=np.float32,
        )
        for record in records
    ])

    fitness = np.array([
        float(
            record.result.get(
                "fitness",
                record.archive_fitness,
            )
        )
        for record in records
    ])

    features = np.array([
        extract_trajectory_features(
            record.result.get(
                "detailed_log",
                []
            )
        )
        for record in records
    ])

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(
        features
    )

    explained = []

    if len(records) >= 2:

        n_components = min(
            10,
            features_scaled.shape[0],
            features_scaled.shape[1],
        )

        if n_components > 0:

            pca = PCA(
                n_components=n_components
            )

            pca.fit(
                features_scaled
            )

            explained = (
                pca.explained_variance_ratio_
                .tolist()
            )

    if len(records) >= max(
        2,
        n_clusters,
    ):

        cluster_score, labels = (
            compute_cluster_separation(
                features_scaled,
                n_clusters=n_clusters,
            )
        )

    else:

        cluster_score = float("nan")

        labels = np.zeros(
            len(records),
            dtype=np.int64,
        )

    if len(records) >= 2:

        smooth_k = min(
            10,
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

    archive_coverage = (
        compute_archive_coverage(
            archive
        )
    )

    report = {

        # MAP-Elites structure metrics
        "archive_coverage": compute_archive_coverage(archive),
        "archive_uniformity": compute_archive_uniformity(archive),
        "occupied_cells": len(archive["grid"]),
        "total_cells": int(np.prod(archive["grid_shape"])),

        # Behavior quality metrics
        "fitness_correlation": fit_corr,
        "local_fitness_smoothness": smoothness,
        "trajectory_diversity": traj_div,

        # Structure of behavior space
        "cluster_silhouette": cluster_score,

        # Dimensionality sanity check
        "pca_explained_variance": explained,
        "pca_2d_total": float(sum(explained[:2])) if explained else 0.0,
    }
    return (
        report,
        bd,
        fitness,
        labels,
    )

def save_outputs(
    output_dir: str,
    report: Dict[str, Any],
    bd: np.ndarray,
    fitness: np.ndarray,
    labels: np.ndarray,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(output_path / "reconstructed_summary.pkl", "wb") as f:
        pickle.dump(
            {
                "bd": bd,
                "fitness": fitness,
                "labels": labels,
            },
            f,
        )

    if bd.shape[1] >= 2:
        plot_bd(bd, fitness, str(output_path / "bd_fitness.png"))
        plot_clusters(bd, labels, str(output_path / "bd_clusters.png"))

def evaluate_controller_no_goal(
    sim: HexapodTorqueSimulation,
    genome: np.ndarray,
) -> dict:
    """
    Evaluate a controller without goal conditioning.
    Returns a single rollout result.
    """

    result = sim.evaluate_controller(
        genome,
        duration=config.SIMULATION_CONFIG["duration"],
        warmup_time=config.SIMULATION_CONFIG["warmup_time"],
        sample_interval=config.SIMULATION_CONFIG["sample_interval"],
        controller_interval=config.SIMULATION_CONFIG["controller_interval"],
        log_all_steps=True,
    )

    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        required=True,
        help="Path to map_elites_final.pkl",
    )
    parser.add_argument(
        "--output",
        default="bd_analysis_from_archive",
        help="Directory where reports and reconstructed trajectories are saved",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional cap on how many archive cells to replay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic replay of the analysis run",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Do not write reconstructed per-cell trajectory pickles to disk",
    )

    args = parser.parse_args()

    archive = load_archive(args.archive)
    _sync_config_from_archive(archive)
    _set_seeds(args.seed if args.seed is not None else archive.get("seed"))

    records = reconstruct_archive_trajectories(
        archive=archive,
        output_dir=args.output,
        max_cells=args.max_cells,
        save_trajectories=not args.no_save_trajectories,
    )

    report, bd, fitness, labels = build_report(
        archive=archive,
        records=records,
        n_clusters=args.clusters,
    )

    report["archive_path"] = os.path.abspath(args.archive)
    report["bd_mode"] = archive.get("bd_mode")
    report["fitness_mode"] = archive.get("fitness_mode")
    report["archive_cells_replayed"] = int(len(records))

    save_outputs(
        output_dir=args.output,
        report=report,
        bd=bd,
        fitness=fitness,
        labels=labels,
    )

    print("=" * 60)
    print("ARCHIVE ANALYSIS")
    print("=" * 60)
    for key, value in report.items():
        if isinstance(value, list):
            continue
        print(f"{key}: {value}")
    print("=" * 60)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()