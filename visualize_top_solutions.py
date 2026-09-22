#!/usr/bin/env python3

"""
Standalone script to visualize and record top MAP-Elites solutions.

Run from terminal:

    python visualize_top_solutions.py

The script opens PyBullet in GUI mode and saves one GIF for each
solution in the top_solutions.json file.
"""

import os
import json
import numpy as np
import time
import sys
import pickle

from config import URDF_PATH
from simulation.simulation_final import HexapodTorqueSimulation

SIMULATION_CONFIG = {
    'duration': 6.0,
    'time_step': 1.0 / 240,
    'gui': False,
    'max_force': 5.0,
    'sample_interval': 1.0,
    'acceleration_threshold': 20.0,
    'warmup_time': 1.0,
    'controller_interval': 4,
}

# ==============================================================
# CONFIGURATION
# ==============================================================

# --------------------------------------------------------------
# Number of solutions to visualize
# --------------------------------------------------------------

MAX_SOLUTIONS = None

# Example:
#
# MAX_SOLUTIONS = 10
#
# None means visualize all solutions in top_solutions.json.


# --------------------------------------------------------------
# GIF settings
# --------------------------------------------------------------

RECORD_FPS = 15

RECORD_WIDTH = 640
RECORD_HEIGHT = 480

# Camera distance from robot
CAMERA_DISTANCE = 2.5

# Camera angle
CAMERA_YAW = 45
CAMERA_PITCH = -25


# --------------------------------------------------------------
# GUI settings
# --------------------------------------------------------------

SHOW_GUI = True


# ==============================================================
# MAIN
# ==============================================================

def main():

    # ==========================================================
    # FIND SOLUTIONS FILE
    # ==========================================================

    EXPERIMENT_DIR = os.path.join(
        os.getcwd(),
        'SaveResults',
        'BalanceVsUnbalance',
        'transformer_latent',
        'transformer',
        'balance_distance_run1'
        #SaveResults/BalanceVsUnbalance/transformer_latent/transformer/balance_distance_run1
    )

    solutions_file = os.path.join(
        EXPERIMENT_DIR,
        'top_solutions.json'
    )

    archive_file = os.path.join(
        os.getcwd(),
        EXPERIMENT_DIR,
        'map_elites_final.pkl'
    )

    def load_or_create_top_solutions(
            solutions_file,
            archive_file,
            num_to_visualize=5
        ):
            """
            Load top_solutions.json if it exists.
    
            If it does not exist, load the final MAP-Elites archive
            from archive_file, extract the grid, find the highest
            fitness solutions, and create top_solutions.json.
            """
    
            # ----------------------------------------------------------
            # Existing JSON
            # ----------------------------------------------------------
    
            if os.path.exists(solutions_file):
    
                print(
                    f"Loading existing solutions from:"
                    f"\n{solutions_file}"
                )
    
                with open(
                    solutions_file,
                    'r'
                ) as f:
    
                    return json.load(f)
    
            # ----------------------------------------------------------
            # No JSON -> load final archive
            # ----------------------------------------------------------
    
            print(
                "\nNo top_solutions.json found."
            )
    
            print(
                "Loading final MAP-Elites archive:"
                f"\n{archive_file}"
            )
    
            if not os.path.exists(
                archive_file
            ):
    
                raise FileNotFoundError(
                    "Neither top_solutions.json nor "
                    "the final archive could be found."
                )
    
            with open(
                archive_file,
                'rb'
            ) as f:
    
                archive = pickle.load(f)
    
            # ----------------------------------------------------------
            # Extract grid
            # ----------------------------------------------------------
    
            if isinstance(
                archive,
                dict
            ):
    
                if 'grid' not in archive:
    
                    raise KeyError(
                        "The loaded archive is a dictionary, "
                        "but does not contain a 'grid' key."
                    )
    
                grid = archive['grid']
    
            elif hasattr(
                archive,
                'grid'
            ):
    
                grid = archive.grid
    
            else:
    
                raise TypeError(
                    "Could not find MAP-Elites grid in the "
                    "loaded pickle."
                )
    
            print(
                f"Found {len(grid)} occupied cells."
            )
    
            # ----------------------------------------------------------
            # Sort by fitness
            # ----------------------------------------------------------
    
            sorted_cells = sorted(
                grid.items(),
                key=lambda x: x[1]['fitness'],
                reverse=True
            )
    
            top_cells = sorted_cells[
                :num_to_visualize
            ]
    
            # ----------------------------------------------------------
            # Create export
            # ----------------------------------------------------------
    
            solutions_data = {
    
                'solutions': [],
    
                'config': {
    
                    'urdf_path':
                        URDF_PATH,
    
                    'duration':
                        SIMULATION_CONFIG['duration'],
    
                    'time_step':
                        SIMULATION_CONFIG['time_step'],
    
                    'sample_interval':
                        SIMULATION_CONFIG['sample_interval'],
    
                    'warmup_time':
                        SIMULATION_CONFIG.get(
                            'warmup_time',
                            1.0
                        ),
    
                    'controller_interval':
                        SIMULATION_CONFIG.get(
                            'controller_interval',
                            4
                        ),
                }
            }
    
            print(
                "\n"
                + "=" * 60
            )
    
            print(
                f"Creating top {len(top_cells)} solutions..."
            )
    
            print(
                "=" * 60
            )
    
            for rank, (
                grid_idx,
                cell
            ) in enumerate(
                top_cells,
                1
            ):
    
                genome = cell['genome']
    
                fitness = cell['fitness']
    
                bd = cell['bd']
    
                print(
                    f"[{rank}] "
                    f"Fitness: {fitness:.4f}, "
                    f"BD: {bd}, "
                    f"Grid: {grid_idx}"
                )
    
                solutions_data[
                    'solutions'
                ].append({
    
                    'rank':
                        rank,
    
                    'grid_idx':
                        grid_idx,
    
                    'fitness':
                        float(fitness),
    
                    'bd':
                        (
                            bd.tolist()
                            if hasattr(
                                bd,
                                'tolist'
                            )
                            else bd
                        ),
    
                    'genome':
                        (
                            genome.tolist()
                            if hasattr(
                                genome,
                                'tolist'
                            )
                            else genome
                        ),
                })
    
            # ----------------------------------------------------------
            # Save JSON
            # ----------------------------------------------------------
    
            with open(
                solutions_file,
                'w'
            ) as f:
    
                json.dump(
                    solutions_data,
                    f,
                    indent=2
                )
    
            print(
                "\nTop solutions exported to:"
                f"\n{solutions_file}"
            )
    
            return solutions_data
    
     

    data = load_or_create_top_solutions(
        solutions_file=solutions_file,
        archive_file=archive_file,
        num_to_visualize=5
    )

    solutions = data['solutions']
    config = data['config']



   # ==============================================================
    # LOAD OR CREATE TOP SOLUTIONS
    # ==============================================================

    # ==========================================================
    # LIMIT NUMBER OF SOLUTIONS
    # ==========================================================

    if MAX_SOLUTIONS is not None:

        solutions = solutions[
            :MAX_SOLUTIONS
        ]

    # ==========================================================
    # URDF
    # ==========================================================

    urdf_path = config.get(
        'urdf_path',
        URDF_PATH
    )

    if not os.path.exists(
        urdf_path
    ):

        print(
            "WARNING: URDF path stored in JSON "
            "does not exist."
        )

        print(
            f"Using configured URDF:"
            f"\n{URDF_PATH}"
        )

        urdf_path = URDF_PATH

    # ==========================================================
    # OUTPUT DIRECTORY
    # ==========================================================

    output_dir = os.path.join(
        os.path.dirname(
            solutions_file
        ),
        'visualizations'
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ==========================================================
    # CONFIGURATION INFORMATION
    # ==========================================================

    print(
        "=" * 70
    )

    print(
        f"Visualizing {len(solutions)} solutions"
    )

    print(
        "=" * 70
    )

    print(
        "\nConfiguration:"
    )

    print(
        f"  Duration: "
        f"{config['duration']}s"
    )

    print(
        f"  Time step: "
        f"{config['time_step']}s"
    )

    expected_steps = int(
        config['duration']
        /
        config['time_step']
    )

    print(
        f"  Expected steps: "
        f"{expected_steps}"
    )

    print(
        f"  GIF FPS: "
        f"{RECORD_FPS}"
    )

    print(
        f"  GIF resolution: "
        f"{RECORD_WIDTH}x{RECORD_HEIGHT}"
    )

    print(
        f"  Output directory:"
        f"\n    {output_dir}"
    )

    print()

    # ==========================================================
    # CREATE SIMULATION
    # ==========================================================

    sim = HexapodTorqueSimulation(
        urdf_path=urdf_path,
        gui=SHOW_GUI,
        time_step=config['time_step'],
    )

    sim.connect()

    # ==========================================================
    # VISUALIZE SOLUTIONS
    # ==========================================================

    for solution_number, sol in enumerate(
        solutions,
        start=1
    ):

        rank = sol['rank']

        fitness = sol['fitness']

        bd = np.array(
            sol['bd']
        )

        genome = np.array(
            sol['genome']
        )

        grid_idx = sol['grid_idx']

        # ======================================================
        # FILENAME
        # ======================================================

        filename = (
            f"rank_{int(rank):03d}"
            f"_fitness_{fitness:.4f}"
            f".gif"
        )

        gif_path = os.path.join(
            output_dir,
            filename
        )

        # ======================================================
        # PRINT INFORMATION
        # ======================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Solution "
            f"{solution_number}/{len(solutions)}"
        )

        print(
            "=" * 70
        )

        print(
            f"Rank:       {rank}"
        )

        print(
            f"Fitness:    {fitness:.6f}"
        )

        print(
            f"BD:         {bd}"
        )

        print(
            f"Grid index: {grid_idx}"
        )

        print(
            f"Genome size: {len(genome)}"
        )

        print(
            f"Output:"
            f"\n  {gif_path}"
        )

        # ======================================================
        # CHECK IF ALREADY EXISTS
        # ======================================================

        if os.path.exists(
            gif_path
        ):

            print(
                "\nGIF already exists."
            )

            answer = input(
                "Overwrite? [y/N]: "
            ).strip().lower()

            if answer != 'y':

                print(
                    "Skipping solution."
                )

                continue

        # ======================================================
        # RUN SIMULATION
        # ======================================================

        print(
            "\nRunning simulation..."
        )

        print(
            "The PyBullet window should show "
            "the robot."
        )

        start_time = time.time()

        try:

            result = sim.evaluate_controller(

                genome,

                duration=
                    config['duration'],

                warmup_time=
                    config.get(
                        'warmup_time',
                        1.0
                    ),

                render=True,

                controller_interval=
                    config.get(
                        'controller_interval',
                        4
                    ),

                log_all_steps=False,

                # ------------------------------------------------
                # RECORDING
                # ------------------------------------------------

                record=True,

                record_path=gif_path,

                record_fps=RECORD_FPS,

                record_width=RECORD_WIDTH,

                record_height=RECORD_HEIGHT,

                record_camera_distance=
                    CAMERA_DISTANCE,

                record_camera_yaw=
                    CAMERA_YAW,

                record_camera_pitch=
                    CAMERA_PITCH,
            )

            elapsed_time = (
                time.time()
                -
                start_time
            )

            # ==================================================
            # RESULTS
            # ==================================================

            print(
                "\nSimulation complete."
            )

            print(
                f"  Distance: "
                f"{result['distance']:.4f}"
            )

            if 'energy' in result:

                print(
                    f"  Energy: "
                    f"{result['energy']:.4f}"
                )

            if 'fitness' in result:

                print(
                    f"  Fitness (re-eval): "
                    f"{result['fitness']:.4f}"
                )

            print(
                f"  Actual elapsed time: "
                f"{elapsed_time:.2f}s"
            )

            print(
                f"  Requested simulation time: "
                f"{config['duration']}s"
            )

            # ==================================================
            # EARLY TERMINATION CHECK
            # ==================================================

            if (
                elapsed_time
                <
                config['duration']
                *
                0.5
            ):

                print(
                    "\n  WARNING:"
                    "\n  Robot likely fell over "
                    "or hit the termination condition."
                )

            elif (
                elapsed_time
                <
                config['duration']
                *
                0.9
            ):

                print(
                    "\n  WARNING:"
                    "\n  Simulation terminated "
                    "before full duration."
                )

            # ==================================================
            # CHECK GIF
            # ==================================================

            if os.path.exists(
                gif_path
            ):

                gif_size_mb = (
                    os.path.getsize(
                        gif_path
                    )
                    /
                    (
                        1024
                        *
                        1024
                    )
                )

                print(
                    f"\n  GIF successfully saved."
                )

                print(
                    f"  File size: "
                    f"{gif_size_mb:.2f} MB"
                )

            else:

                print(
                    "\n  WARNING:"
                    "\n  GIF file was not created."
                )

        except Exception as e:

            print(
                "\n  ERROR during simulation:"
            )

            print(
                f"  {e}"
            )

        # ======================================================
        # WAIT BETWEEN SOLUTIONS
        # ======================================================

        if (
            solution_number
            <
            len(solutions)
        ):

            print(
                "\nMoving to next solution..."
            )

            # Small pause to allow the GUI
            # to update before resetting.
            time.sleep(0.5)

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Visualization complete."
    )

    print(
        "=" * 70
    )

    print(
        f"\nGIFs saved in:"
        f"\n{output_dir}"
    )

    print()

    # ==========================================================
    # PYBULLET TEARDOWN
    # ==========================================================
    #
    # In this environment, PyBullet GUI teardown can
    # intermittently segfault during interpreter shutdown.
    # Exit immediately to avoid that path.
    # ==========================================================

    sys.stdout.flush()
    sys.stderr.flush()

    os._exit(0)


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == '__main__':

    main()