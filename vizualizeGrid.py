#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


ARCHIVE_PATH = "SaveResults/Goal/GoalTrans/map_elites_final.pkl"


def main():

    with open(ARCHIVE_PATH, "rb") as f:
        archive = pickle.load(f)

    print(type(archive))

    if isinstance(archive, dict):
        print("\nKeys:")
        print(archive.keys())

    print("=" * 80)
    print("MAP-Elites Grid Information")
    print("=" * 80)

    print("\nBD dimensions:")
    print(archive['bd_dims'])

    print("\nNumber of BD dimensions:")
    print(len(archive['bd_dims']))

    print("\nFilled cells:")
    print(len(archive['grid']))

    print()

    for dim_idx, dim in enumerate(archive['bd_dims']):

        print("-" * 60)
        print(f"Dimension {dim_idx}")

        if isinstance(dim, tuple) and len(dim) == 3:
            low, high, bins = dim

            print(f"Range : [{low}, {high}]")
            print(f"Bins  : {bins}")
            print(f"Width : {(high - low) / bins}")

        else:
            print(dim)

    print("\n" + "=" * 80)
    print("Occupancy per dimension")
    print("=" * 80)

    occupied_indices = np.array(list(archive['grid'].keys()))

    for dim_idx in range(occupied_indices.shape[1]):

        unique_bins = np.unique(
            occupied_indices[:, dim_idx]
        )

        print(
            f"Dimension {dim_idx}: "
            f"{len(unique_bins)} occupied bins"
        )

        print(
            f"Occupied bin indices: "
            f"{unique_bins}"
        )

        print()



    occupied = np.array(list(archive["grid"].keys()))

    print("Shape:", occupied.shape)

    # variance per dimension
    print("\nSTD per dimension:")
    print(np.std(occupied, axis=0))

    print("\nUnique bins per dimension:")
    for i in range(occupied.shape[1]):
        print(i, len(np.unique(occupied[:, i])))

        print("\nGenerating occupancy plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    dim_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
    ]

    for ax, (d1, d2) in zip(axes, dim_pairs):

        bins_x = archive["bd_dims"][d1][2]
        bins_y = archive["bd_dims"][d2][2]

        occupancy = np.zeros((bins_y, bins_x))

        for idx in archive["grid"].keys():
            occupancy[idx[d2], idx[d1]] += 1

        masked = np.ma.masked_where(
            occupancy == 0,
            occupancy
        )

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")

        ax.imshow(
            masked,
            origin="lower",
            interpolation="nearest",
            cmap=cmap
        )

        ax.set_title(f"BD dims {d1} vs {d2}")
        ax.set_xlabel(f"Dim {d1}")
        ax.set_ylabel(f"Dim {d2}")

    plt.tight_layout()

    output_file = "grid_occupancy.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved grid visualization to: {output_file}")

    plt.close()


if __name__ == "__main__":
    main()