"""
Cluster saved random trajectories and color BD points by cluster.

This script loads trajectory files from:
- SaveResults/finalXY/random_trajectories
- SaveResults/finalTransformer/random_trajectories

For each dataset, it:
1. Loads saved detailed trajectories and saved behavior descriptors (BDs).
2. Builds trajectory feature vectors from detailed logs.
3. Clusters trajectories in feature space.
4. Plots saved BD points colored by cluster labels.

Important:
- BDs are read from the trajectory files and are not recomputed.
"""

import argparse
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SavedTrajectory:
    file_path: str
    bd: np.ndarray
    detailed_log: List[dict]


def _to_1d_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _get_field(step: dict, names: List[str], size: int) -> np.ndarray:
    for name in names:
        if name in step and step[name] is not None:
            arr = _to_1d_array(step[name])
            if arr is None:
                continue
            if arr.size < size:
                return np.concatenate([arr, np.zeros(size - arr.size, dtype=np.float32)])
            if arr.size > size:
                return arr[:size]
            return arr
    return np.zeros(size, dtype=np.float32)


def _infer_dims(records: List[SavedTrajectory]) -> Dict[str, int]:
    defaults = {
        "position": 3,
        "orientation": 4,
        "contacts": 6,
        "torques": 18,
    }

    for r in records:
        if not r.detailed_log:
            continue
        step = r.detailed_log[0]
        return {
            "position": int(np.asarray(step.get("position", np.zeros(defaults["position"]))).size),
            "orientation": int(np.asarray(step.get("orientation", np.zeros(defaults["orientation"]))).size),
            "contacts": int(np.asarray(step.get("contacts", np.zeros(defaults["contacts"]))).size),
            "torques": int(np.asarray(step.get("torques", np.zeros(defaults["torques"]))).size),
        }

    return defaults


def _step_vector(step: dict, dims: Dict[str, int]) -> np.ndarray:
    parts = [
        _get_field(step, ["position", "body_position", "base_position"], dims["position"]),
        _get_field(step, ["orientation", "body_orientation", "base_orientation"], dims["orientation"]),
        _get_field(step, ["contacts", "contact_states", "foot_contacts"], dims["contacts"]),
        _get_field(step, ["torques", "joint_torques"], dims["torques"]),
    ]
    return np.concatenate(parts).astype(np.float32)


def _downsample(sequence: np.ndarray, target_len: int) -> np.ndarray:
    if len(sequence) == 0:
        raise ValueError("Cannot downsample an empty sequence")
    if len(sequence) == target_len:
        return sequence
    idx = np.linspace(0, len(sequence) - 1, target_len).astype(int)
    return sequence[idx]


def _trajectory_features(records: List[SavedTrajectory], downsample_len: int) -> np.ndarray:
    dims = _infer_dims(records)
    feats = []
    for r in records:
        seq = np.array([_step_vector(s, dims) for s in r.detailed_log], dtype=np.float32)
        sampled = _downsample(seq, downsample_len)
        feats.append(sampled.reshape(-1))
    return np.array(feats, dtype=np.float32)


def _resolve_entry(data: dict) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
    payload = data.get("result", data)

    bd = data.get("behavior_descriptors", None)
    if bd is None:
        bd = data.get("bd", None)
    if bd is None and isinstance(payload, dict):
        bd = payload.get("behavior_descriptors", None)
    if bd is None and isinstance(payload, dict):
        bd = payload.get("bd", None)

    if bd is None:
        return None, None

    bd_arr = _to_1d_array(bd)
    if bd_arr is None or bd_arr.size < 2:
        return None, None

    detailed_log = None
    if "detailed_log" in data and isinstance(data["detailed_log"], list):
        detailed_log = data["detailed_log"]
    elif isinstance(payload, dict) and "detailed_log" in payload and isinstance(payload["detailed_log"], list):
        detailed_log = payload["detailed_log"]

    if not detailed_log:
        return None, None

    return bd_arr, detailed_log


def load_saved_trajectories(folder: str, pattern: str, max_trajectories: Optional[int]) -> List[SavedTrajectory]:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern {pattern}")

    records: List[SavedTrajectory] = []
    for fp in files:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            continue

        bd, detailed_log = _resolve_entry(data)
        if bd is None or detailed_log is None:
            continue

        records.append(SavedTrajectory(file_path=fp, bd=bd, detailed_log=detailed_log))

        if max_trajectories is not None and len(records) >= max_trajectories:
            break

    if not records:
        raise ValueError(f"No valid trajectory records with saved BD and detailed_log in {folder}")

    return records


def _pick_k(features: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    n = features.shape[0]
    k_max = min(k_max, max(2, n - 1))
    if n < 3:
        return 1

    best_k = 2
    best_score = -np.inf

    for k in range(max(2, k_min), k_max + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(features)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_features(
    features_scaled: np.ndarray,
    method: str,
    n_clusters: int,
    k_min: int,
    k_max: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    if method == "dbscan":
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(features_scaled)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, int(k)

    if n_clusters <= 0:
        n_clusters = _pick_k(features_scaled, k_min, k_max, seed)

    if method == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features_scaled)
    else:
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(features_scaled)

    return labels, int(n_clusters)


def plot_bd_clusters(bd_values: np.ndarray, labels: np.ndarray, out_path: str, title: str):
    plot_values = bd_values[:, :2] if bd_values.shape[1] > 2 else bd_values

    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique):
        mask = labels == label
        label_name = "noise" if label == -1 else f"cluster {int(label)}"
        plt.scatter(
            plot_values[mask, 0],
            plot_values[mask, 1],
            s=18,
            alpha=0.8,
            c=[cmap(i)],
            label=label_name,
        )

    plt.xlabel("BD dim 0")
    plt.ylabel("BD dim 1")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_pca(features_scaled: np.ndarray, labels: np.ndarray, out_path: str, title: str):
    pca = PCA(n_components=2)
    emb = pca.fit_transform(features_scaled)
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(unique):
        mask = labels == label
        label_name = "noise" if label == -1 else f"cluster {int(label)}"
        plt.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.8, c=[cmap(i)], label=label_name)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_dataset(
    name: str,
    logs_folder: str,
    pattern: str,
    output_dir: str,
    max_trajectories: Optional[int],
    downsample_len: int,
    method: str,
    n_clusters: int,
    k_min: int,
    k_max: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    seed: int,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    records = load_saved_trajectories(logs_folder, pattern, max_trajectories)
    bd_values = np.array([r.bd for r in records], dtype=np.float32)

    feat = _trajectory_features(records, downsample_len)
    feat_scaled = StandardScaler().fit_transform(feat)

    labels, k = cluster_features(
        features_scaled=feat_scaled,
        method=method,
        n_clusters=n_clusters,
        k_min=k_min,
        k_max=k_max,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        seed=seed,
    )

    plot_feature_pca(
        feat_scaled,
        labels,
        os.path.join(output_dir, "trajectory_clusters_pca.png"),
        f"{name} trajectory clusters (PCA)",
    )
    plot_bd_clusters(
        bd_values,
        labels,
        os.path.join(output_dir, "saved_bd_colored_by_cluster.png"),
        f"{name} saved BDs colored by trajectory clusters",
    )

    report = {
        "dataset": name,
        "logs_folder": logs_folder,
        "num_trajectories": int(len(records)),
        "bd_dim": int(bd_values.shape[1]),
        "cluster_method": method,
        "n_clusters": int(k),
        "cluster_sizes": {
            str(int(label)): int(np.sum(labels == label))
            for label in np.unique(labels)
        },
        "used_saved_bds": True,
        "bd_recomputed": False,
    }

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def plot_combined_comparison(
    xy_bd: np.ndarray,
    xy_labels: np.ndarray,
    tr_bd: np.ndarray,
    tr_labels: np.ndarray,
    out_path: str,
):
    xy_plot = xy_bd[:, :2] if xy_bd.shape[1] > 2 else xy_bd
    tr_plot = tr_bd[:, :2] if tr_bd.shape[1] > 2 else tr_bd

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, labels, title in [
        (axes[0], xy_plot, xy_labels, "finalXY: saved BDs colored by clusters"),
        (axes[1], tr_plot, tr_labels, "finalTransformer: saved BDs colored by clusters"),
    ]:
        unique = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
        for i, label in enumerate(unique):
            mask = labels == label
            label_name = "noise" if label == -1 else f"cluster {int(label)}"
            ax.scatter(data[mask, 0], data[mask, 1], s=14, alpha=0.8, c=[cmap(i)], label=label_name)
        ax.set_xlabel("BD dim 0")
        ax.set_ylabel("BD dim 1")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster random trajectories from finalXY/finalTransformer and color saved BD points by cluster."
    )
    parser.add_argument(
        "--xy-folder",
        default=os.path.join("SaveResults", "finalXY", "random_trajectories"),
    )
    parser.add_argument(
        "--transformer-folder",
        default=os.path.join("SaveResults", "finalTransformer", "random_trajectories"),
    )
    parser.add_argument("--pattern", default="traj_*.pkl")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("SaveResults", "final_descriptor_cluster_viz"),
    )
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--downsample-len", type=int, default=60)

    parser.add_argument("--cluster-method", choices=["kmeans", "agglomerative", "dbscan"], default="dbscan")
    parser.add_argument("--clusters", type=int, default=0, help="0 enables automatic selection for kmeans/agglomerative")
    parser.add_argument("--cluster-min", type=int, default=2)
    parser.add_argument("--cluster-max", type=int, default=10)
    parser.add_argument("--dbscan-eps", type=float, default=0.8)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process XY
    xy_records = load_saved_trajectories(args.xy_folder, args.pattern, args.max_trajectories)
    xy_bd = np.array([r.bd for r in xy_records], dtype=np.float32)
    xy_feat = _trajectory_features(xy_records, args.downsample_len)
    xy_scaled = StandardScaler().fit_transform(xy_feat)
    xy_labels, xy_k = cluster_features(
        features_scaled=xy_scaled,
        method=args.cluster_method,
        n_clusters=args.clusters,
        k_min=args.cluster_min,
        k_max=args.cluster_max,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        seed=args.seed,
    )

    xy_dir = os.path.join(args.output_dir, "finalXY")
    os.makedirs(xy_dir, exist_ok=True)
    plot_feature_pca(
        xy_scaled,
        xy_labels,
        os.path.join(xy_dir, "trajectory_clusters_pca.png"),
        "finalXY trajectory clusters (PCA)",
    )
    plot_bd_clusters(
        xy_bd,
        xy_labels,
        os.path.join(xy_dir, "saved_bd_colored_by_cluster.png"),
        "finalXY saved BDs colored by trajectory clusters",
    )

    # Load and process Transformer
    tr_records = load_saved_trajectories(args.transformer_folder, args.pattern, args.max_trajectories)
    tr_bd = np.array([r.bd for r in tr_records], dtype=np.float32)
    tr_feat = _trajectory_features(tr_records, args.downsample_len)
    tr_scaled = StandardScaler().fit_transform(tr_feat)
    tr_labels, tr_k = cluster_features(
        features_scaled=tr_scaled,
        method=args.cluster_method,
        n_clusters=args.clusters,
        k_min=args.cluster_min,
        k_max=args.cluster_max,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        seed=args.seed,
    )

    tr_dir = os.path.join(args.output_dir, "finalTransformer")
    os.makedirs(tr_dir, exist_ok=True)
    plot_feature_pca(
        tr_scaled,
        tr_labels,
        os.path.join(tr_dir, "trajectory_clusters_pca.png"),
        "finalTransformer trajectory clusters (PCA)",
    )
    plot_bd_clusters(
        tr_bd,
        tr_labels,
        os.path.join(tr_dir, "saved_bd_colored_by_cluster.png"),
        "finalTransformer saved BDs colored by trajectory clusters",
    )

    plot_combined_comparison(
        xy_bd=xy_bd,
        xy_labels=xy_labels,
        tr_bd=tr_bd,
        tr_labels=tr_labels,
        out_path=os.path.join(args.output_dir, "combined_saved_bd_colored_by_cluster.png"),
    )

    summary = {
        "finalXY": {
            "num_trajectories": int(len(xy_records)),
            "bd_dim": int(xy_bd.shape[1]),
            "cluster_method": args.cluster_method,
            "n_clusters": int(xy_k),
            "cluster_sizes": {str(int(k)): int(np.sum(xy_labels == k)) for k in np.unique(xy_labels)},
            "used_saved_bds": True,
            "bd_recomputed": False,
        },
        "finalTransformer": {
            "num_trajectories": int(len(tr_records)),
            "bd_dim": int(tr_bd.shape[1]),
            "cluster_method": args.cluster_method,
            "n_clusters": int(tr_k),
            "cluster_sizes": {str(int(k)): int(np.sum(tr_labels == k)) for k in np.unique(tr_labels)},
            "used_saved_bds": True,
            "bd_recomputed": False,
        },
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Saved trajectory clustering + BD coloring complete")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"finalXY trajectories: {len(xy_records)} | clusters: {xy_k}")
    print(f"finalTransformer trajectories: {len(tr_records)} | clusters: {tr_k}")
    print("BDs were loaded from files and not recomputed.")


if __name__ == "__main__":
    main()
