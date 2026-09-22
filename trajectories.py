#!/usr/bin/env python3

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

SAVE_DIR = Path("SaveResults/Goal/Goal8DTrans")

TRAJECTORY_DIR = Path(
    SAVE_DIR / "bd_analysis_from_archive/reconstructed_trajectories"
)


# Number of samples every trajectory is resampled to
N_SAMPLES = 100

# Plot at most this many individual trajectories
MAX_TRAJECTORIES_TO_DRAW = 500

# Optional goal position
GOAL = None
# Example:
# GOAL = (2.0, 1.5)


# ----------------------------------------------------------------------
# LOAD ONE TRAJECTORY
# ----------------------------------------------------------------------

def load_xy_trajectory(pkl_file):
    """
    Returns:
        x : np.ndarray
        y : np.ndarray
    """

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    log = data["result"]["detailed_log"]

    if len(log) == 0:
        print(f"{pkl_file} has an empty detailed_log")
        return np.array([]), np.array([])

    #print(type(data))
    #print(data.keys())

    #print(type(log))
    #print(len(log))

    #print(log[0])

    x = []
    y = []

    for step in log:

        pos = step["position"]

        x.append(pos[0])
        y.append(pos[1])

    return np.asarray(x), np.asarray(y)


# ----------------------------------------------------------------------
# RESAMPLE
# ----------------------------------------------------------------------

def resample(arr, n_samples):
    old_t = np.linspace(0, 1, len(arr))
    new_t = np.linspace(0, 1, n_samples)

    return np.interp(new_t, old_t, arr)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

TOP_K = 20

files = sorted(TRAJECTORY_DIR.glob("cell_*.pkl"))[:TOP_K]

print(f"Using top {len(files)} trajectories")

all_x = []
all_y = []

for i, file in enumerate(files):

    if i % 500 == 0:
        print(f"{i}/{len(files)}")

    x, y = load_xy_trajectory(file)

    if len(x) == 0 or len(y) == 0:
        print(f"Skipping {file.name} (empty trajectory)")
        continue

    x = resample(x, N_SAMPLES)
    y = resample(y, N_SAMPLES)

    all_x.append(x)
    all_y.append(y)

all_x = np.stack(all_x)
all_y = np.stack(all_y)

mean_x = np.mean(all_x, axis=0)
mean_y = np.mean(all_y, axis=0)

std_x = np.std(all_x, axis=0)
std_y = np.std(all_y, axis=0)

print("Creating figure...")

plt.figure(figsize=(8, 8))

# ----------------------------------------------------------------------
# Draw a subset of trajectories
# ----------------------------------------------------------------------



cmap = plt.get_cmap("tab20")

step = max(1, len(all_x) // MAX_TRAJECTORIES_TO_DRAW)

trajectory_indices = range(0, len(all_x), step)

for j, i in enumerate(trajectory_indices):
    plt.plot(
        all_x[i],
        all_y[i],
        color=cmap(j % 20),
        linewidth=1.8,
        alpha=0.9,
    )
# ----------------------------------------------------------------------
# Mean trajectory
# ----------------------------------------------------------------------

#plt.plot( mean_x, mean_y, color="red", linewidth=1, label="Mean trajectory",)

# ----------------------------------------------------------------------
# Start / End
# ----------------------------------------------------------------------

plt.scatter(
    mean_x[0],
    mean_y[0],
    s=20,
    color="green",
    zorder=10,
    label="Start",
)

#plt.scatter(
#    mean_x[-1],
#    mean_y[-1],
#    s=120,
#    color="red",
#    edgecolor="black",
#    zorder=10,
#    label="End",
#)

# ----------------------------------------------------------------------
# Goal
# ----------------------------------------------------------------------

if GOAL is not None:
    plt.scatter(
        GOAL[0],
        GOAL[1],
        marker="*",
        s=250,
        color="gold",
        edgecolor="black",
        label="Goal",
    )

# ----------------------------------------------------------------------
# Cosmetics
# ----------------------------------------------------------------------

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 0.5)

ax = plt.gca()
ax.set_aspect("equal")

plt.grid(True, alpha=0.3)

plt.xlabel("X Position (m)", fontsize=18)
plt.ylabel("Y Position (m)", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title("Robot trajectories", fontsize=20)

plt.legend()

plt.tight_layout()

plt.savefig(
    SAVE_DIR / "trajectory_overview.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved figure as trajectory_overview.png")