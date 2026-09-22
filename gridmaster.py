#!/usr/bin/env python3

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

ARCHIVE_FILE = Path(
    "SaveResults/5000Runs/5000XY/map_elites_final.pkl"
)

OUTPUT_FILE = ARCHIVE_FILE.parent / "final_grid.png"

# MAP-Elites grid is ALWAYS 50 x 50
GRID_SIZE = 50

# BD ranges
BD1_MIN = 0
BD1_MAX = 50

BD2_MIN = 0
BD2_MAX = 50

FIGSIZE = (9, 8)

COLORMAP = "viridis"

TICK_FONTSIZE = 14
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20


# ======================================================================
# LOAD ARCHIVE
# ======================================================================

def load_archive(path):

    print(f"Loading archive:")
    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Archive not found:\n{path}"
        )

    with open(path, "rb") as f:
        archive = pickle.load(f)

    print("Archive loaded.")

    return archive


# ======================================================================
# EXTRACT GRID
# ======================================================================

def extract_grid(archive):

    if isinstance(archive, dict):

        if "grid" not in archive:
            raise KeyError(
                "Archive dictionary does not contain 'grid'."
            )

        return archive["grid"]

    if hasattr(archive, "grid"):
        return archive.grid

    raise TypeError(
        f"Could not find grid in archive of type {type(archive)}"
    )


# ======================================================================
# GET CELL FITNESS
# ======================================================================

def get_fitness(cell):

    if isinstance(cell, dict):
        return float(cell["fitness"])

    if hasattr(cell, "fitness"):
        return float(cell.fitness)

    raise KeyError(
        f"Could not find fitness in cell of type {type(cell)}"
    )


# ======================================================================
# PLOT FINAL GRID
# ======================================================================

def plot_grid(grid):

    print(f"Occupied cells: {len(grid)}")
    print(f"Grid size: {GRID_SIZE} x {GRID_SIZE}")

    # --------------------------------------------------------------
    # Create empty 50 x 50 grid
    #
    # NaN = empty cell
    # --------------------------------------------------------------

    fitness_grid = np.full(
        (GRID_SIZE, GRID_SIZE),
        np.nan
    )

    # --------------------------------------------------------------
    # Fill grid with fitness values
    # --------------------------------------------------------------

    for grid_idx, cell in grid.items():

        x_idx = int(grid_idx[0])
        y_idx = int(grid_idx[1])

        # Safety check
        if not (0 <= x_idx < GRID_SIZE):
            continue

        if not (0 <= y_idx < GRID_SIZE):
            continue

        fitness_grid[x_idx, y_idx] = get_fitness(cell)

    # --------------------------------------------------------------
    # Mask empty cells
    # --------------------------------------------------------------

    masked_grid = np.ma.masked_invalid(
        fitness_grid
    )

    # --------------------------------------------------------------
    # Create figure
    # --------------------------------------------------------------

    print("Creating figure...")

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------------
    # Plot the actual 50 x 50 grid
    # --------------------------------------------------------------

    image = ax.imshow(
        masked_grid,
        origin="lower",

        # Make the image cover the actual BD range
        extent=[
            BD1_MIN,
            BD1_MAX,
            BD2_MIN,
            BD2_MAX
        ],

        cmap=COLORMAP,
        interpolation="none",
        aspect="equal"
    )

    # --------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------

    cbar = fig.colorbar(
        image,
        ax=ax
    )

    cbar.set_label(
        "Fitness",
        fontsize=LABEL_FONTSIZE
    )

    cbar.ax.tick_params(
        labelsize=TICK_FONTSIZE
    )

    # --------------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------------

    ax.set_xlabel(
        "X position",
        fontsize=LABEL_FONTSIZE
    )

    ax.set_ylabel(
        "Y position",
        fontsize=LABEL_FONTSIZE
    )

    ax.set_title(
        "Final MAP-Elites Archive",
        fontsize=TITLE_FONTSIZE
    )

    # --------------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------------

    ax.set_xlim(
        BD1_MIN,
        BD1_MAX
    )

    ax.set_ylim(
        BD2_MIN,
        BD2_MAX
    )

    # --------------------------------------------------------------
    # Tick labels
    # --------------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE
    )

    # --------------------------------------------------------------
    # Draw the 50 x 50 cell boundaries
    # --------------------------------------------------------------

    cell_width = (
        BD1_MAX - BD1_MIN
    ) / GRID_SIZE

    cell_height = (
        BD2_MAX - BD2_MIN
    ) / GRID_SIZE

    # Minor ticks at cell boundaries
    ax.set_xticks(
        np.linspace(
            BD1_MIN,
            BD1_MAX,
            GRID_SIZE + 1
        ),
        minor=True
    )

    ax.set_yticks(
        np.linspace(
            BD2_MIN,
            BD2_MAX,
            GRID_SIZE + 1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        linewidth=0.3,
        alpha=0.4
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # --------------------------------------------------------------
    # Major ticks
    # --------------------------------------------------------------

    ax.set_xticks(
        np.arange(
            BD1_MIN,
            BD1_MAX + 1,
            10
        )
    )

    ax.set_yticks(
        np.arange(
            BD2_MIN,
            BD2_MAX + 1,
            10
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    plt.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print(f"Saved figure:")
    print(f"  {OUTPUT_FILE}")

    plt.show()


# ======================================================================
# MAIN
# ======================================================================

def main():

    archive = load_archive(
        ARCHIVE_FILE
    )

    grid = extract_grid(
        archive
    )

    plot_grid(
        grid
    )


if __name__ == "__main__":
    main()