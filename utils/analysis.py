"""
Analyze the quality of behavior descriptors (BDs) using trajectory data.

This script evaluates whether a BD space is meaningful by measuring:

1. Coverage
   - How much of the BD space is occupied

2. Uniformity
   - Whether solutions are evenly distributed

3. Fitness Correlation
   - Whether nearby BDs have similar fitness

4. Cluster Separation
   - Whether trajectory clusters separate in BD space

5. Local Smoothness
   - Whether nearby points behave similarly

6. Trajectory Diversity
   - Diversity in trajectory feature space

7. PCA Explained Variance
   - How much trajectory information is preserved

The script:
- loads saved trajectory logs
- extracts trajectory features
- clusters trajectories
- evaluates BD quality metrics
- generates plots
- saves reports

Works for:
- XY descriptors
- Transformer descriptors
- Any descriptor dimension
"""

import argparse
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ==========================================================
# DATA STRUCTURE
# ==========================================================

@dataclass
class SavedTrajectory:
    bd: np.ndarray
    fitness: float
    detailed_log: List[dict]


# ==========================================================
# LOAD DATA
# ==========================================================

def load_saved_trajectories(folder, pattern="*.pkl", max_trajectories=None, seed=42):

    files = sorted(
        glob.glob(os.path.join(folder, pattern))
    )

    if max_trajectories is not None and max_trajectories < len(files):
        rng = np.random.default_rng(seed)
        files = list(rng.permutation(files))

    records = []

    for fp in files:

        with open(fp, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            continue

        payload = data
        result = data.get("result")
        if isinstance(result, dict):
            payload = result

        if "behavior_descriptors" not in payload:
            continue

        if "detailed_log" not in payload:
            continue

        records.append(
            SavedTrajectory(
                bd=np.asarray(
                    payload["behavior_descriptors"],
                    dtype=np.float32
                ),
                fitness=float(
                    payload.get("fitness", data.get("fitness", 0.0))
                ),
                detailed_log=payload["detailed_log"],
            )
        )

        if max_trajectories is not None and len(records) >= max_trajectories:
            break

    return records


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_trajectory_features(
    detailed_log,
    target_len=50
):

    seq = []

    for step in detailed_log:

        position = np.asarray(
            step.get("position", [0, 0, 0]),
            dtype=np.float32
        )

        orientation = np.asarray(
            step.get("orientation", [0, 0, 0, 1]),
            dtype=np.float32
        )

        contacts = np.asarray(
            step.get("contacts", np.zeros(6)),
            dtype=np.float32
        )

        torques = np.asarray(
            step.get("torques", np.zeros(18)),
            dtype=np.float32
        )

        vec = np.concatenate([
            position,
            orientation,
            contacts,
            torques
        ])

        seq.append(vec)

    seq = np.asarray(seq)

    if len(seq) == 0:
        return np.zeros(target_len * 31)

    idx = np.linspace(
        0,
        len(seq) - 1,
        target_len
    ).astype(int)

    seq = seq[idx]

    return seq.flatten()


# ==========================================================
# COVERAGE
# ==========================================================

def _digitize_behavior_descriptors(bd, bins=50):
    bd = np.asarray(bd, dtype=np.float64)

    if bd.ndim != 2:
        raise ValueError("Behavior descriptors must be a 2D array")

    num_dims = bd.shape[1]

    if np.isscalar(bins):
        bin_counts = [int(bins)] * num_dims
    else:
        bin_counts = [int(b) for b in bins]
        if len(bin_counts) != num_dims:
            raise ValueError("bins must have one entry per BD dimension")

    digitized = []

    for dim in range(num_dims):
        values = bd[:, dim]
        dim_bins = bin_counts[dim]

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            digitized.append(np.zeros(len(values), dtype=np.int64))
            continue

        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))

        if np.isclose(min_value, max_value):
            digitized.append(np.zeros(len(values), dtype=np.int64))
            bin_counts[dim] = 1
            continue

        edges = np.linspace(min_value, max_value, dim_bins + 1)
        indices = np.digitize(values, edges[1:-1], right=False)
        indices = np.clip(indices, 0, dim_bins - 1)
        digitized.append(indices.astype(np.int64))

    return np.stack(digitized, axis=1), bin_counts

def compute_coverage(
    bd,
    bins=50
):

    digitized, bin_counts = _digitize_behavior_descriptors(bd, bins=bins)

    occupied = np.unique(digitized, axis=0).shape[0]
    total = int(np.prod(bin_counts, dtype=object))

    return occupied / total if total > 0 else 0.0


# ==========================================================
# UNIFORMITY
# ==========================================================

def compute_uniformity(
    bd,
    bins=50
):

    digitized, _ = _digitize_behavior_descriptors(bd, bins=bins)

    _, counts = np.unique(digitized, axis=0, return_counts=True)

    p = counts.astype(np.float64)

    p = p / np.sum(p)

    entropy = -np.sum(
        p * np.log(p + 1e-12)
    )

    max_entropy = np.log(len(p))

    return entropy / max_entropy


# ==========================================================
# LOCAL FITNESS SMOOTHNESS
# ==========================================================

def compute_local_fitness_smoothness(
    bd,
    fitness,
    k=10
):

    nbrs = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(bd)

    distances, indices = nbrs.kneighbors(bd)

    diffs = []

    for i in range(len(bd)):

        neighbors = indices[i][1:]

        local_diff = np.mean(
            np.abs(
                fitness[i] - fitness[neighbors]
            )
        )

        diffs.append(local_diff)

    return float(np.mean(diffs))


# ==========================================================
# CLUSTER SEPARATION
# ==========================================================

def compute_cluster_separation(
    features,
    n_clusters=6
):

    labels = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=42
    ).fit_predict(features)

    score = silhouette_score(
        features,
        labels
    )

    return score, labels


# ==========================================================
# FITNESS CORRELATION
# ==========================================================

def compute_fitness_correlation(
    bd,
    fitness
):

    d_bd = pairwise_distances(bd)

    d_fit = pairwise_distances(
        fitness.reshape(-1, 1)
    )

    corr = np.corrcoef(
        d_bd.flatten(),
        d_fit.flatten()
    )[0, 1]

    return corr


# ==========================================================
# TRAJECTORY DIVERSITY
# ==========================================================

def compute_trajectory_diversity(
    features
):

    d = pairwise_distances(features)

    return np.mean(d)


# ==========================================================
# PLOTS
# ==========================================================

def plot_bd(
    bd,
    fitness,
    out_path
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        bd[:, 0],
        bd[:, 1],
        c=fitness,
        s=10,
    )

    plt.colorbar(label="Fitness")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("BD 1", fontsize=18)
    plt.ylabel("BD 2", fontsize=18)

    plt.title("Behavior Descriptor Space", fontsize=20)

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()


def plot_clusters(
    bd,
    labels,
    out_path
):

    plt.figure(figsize=(8, 6))

    plt.xlim(-3, 1)
    plt.ylim(-1.5, 2)

    plt.scatter(
        bd[:, 0],
        bd[:, 1],
        c=labels,
        s=10,
    )

    plt.xlabel("BD 1", fontsize=18)
    plt.ylabel("BD 2", fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title("Trajectory Clusters in BD Space", fontsize=20)

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def analyze_dataset(
    folder,
    output_dir,
    bins=50,
    n_clusters=6,
    num_trajectories=10000,
    seed=42
):

    os.makedirs(output_dir, exist_ok=True)

    records = load_saved_trajectories(
        folder,
        max_trajectories=num_trajectories,
        seed=seed,
    )

    print(
        f"Loaded {len(records)} trajectories from {folder} "
        f"(requested {num_trajectories})"
    )

    if len(records) == 0:
        raise ValueError(
            f"No valid trajectory files found in {folder}"
        )

    if len(records) < num_trajectories:
        print(
            f"Warning: only {len(records)} valid trajectories were available"
        )

    bd = np.array([
        r.bd for r in records
    ])

    fitness = np.array([
        r.fitness for r in records
    ])

    features = np.array([
        extract_trajectory_features(
            r.detailed_log
        )
        for r in records
    ])

    # ======================================================
    # STANDARDIZE FEATURES
    # ======================================================

    scaler = StandardScaler()

    features_scaled = scaler.fit_transform(
        features
    )

    # ======================================================
    # PCA ANALYSIS
    # ======================================================

    pca = PCA(n_components=10)

    pca.fit(features_scaled)

    explained = pca.explained_variance_ratio_

    # ======================================================
    # CLUSTERING
    # ======================================================

    cluster_score, labels = (
        compute_cluster_separation(
            features_scaled,
            n_clusters=n_clusters
        )
    )

    # ======================================================
    # METRICS
    # ======================================================

    coverage = compute_coverage(
        bd,
        bins=bins
    )

    uniformity = compute_uniformity(
        bd,
        bins=bins
    )

    smoothness = (
        compute_local_fitness_smoothness(
            bd,
            fitness
        )
    )

    fit_corr = compute_fitness_correlation(
        bd,
        fitness
    )

    traj_div = compute_trajectory_diversity(
        features_scaled
    )

    # ======================================================
    # SAVE REPORT
    # ======================================================

    report = {

        "num_trajectories":
            int(len(records)),

        "bd_dim":
            int(bd.shape[1]),

        "coverage":
            float(coverage),

        "uniformity":
            float(uniformity),

        "fitness_correlation":
            float(fit_corr),

        "local_fitness_smoothness":
            float(smoothness),

        "trajectory_diversity":
            float(traj_div),

        "cluster_silhouette":
            float(cluster_score),

        "pca_explained_variance":
            explained.tolist(),

        "pca_2d_total":
            float(explained[:2].sum()),
    }

    with open(
        os.path.join(output_dir, "report.json"),
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ======================================================
    # PLOTS
    # ======================================================

    plot_bd(
        bd,
        fitness,
        os.path.join(
            output_dir,
            "bd_fitness.png"
        )
    )

    plot_clusters(
        bd,
        labels,
        os.path.join(
            output_dir,
            "bd_clusters.png"
        )
    )

    # ======================================================
    # PRINT SUMMARY
    # ======================================================

    print("=" * 60)
    print("BD QUALITY REPORT")
    print("=" * 60)

    for k, v in report.items():

        if isinstance(v, list):
            continue

        print(f"{k}: {v}")

    print("=" * 60)

    print(f"Results saved to: {output_dir}")


# ==========================================================
# ENTRY
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="bd_analysis"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=6
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1000,
        help="Number of random trajectories to sample from the folder"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling trajectories"
    )

    args = parser.parse_args()

    analyze_dataset(
        folder=args.folder,
        output_dir=args.output,
        bins=args.bins,
        n_clusters=args.clusters,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()