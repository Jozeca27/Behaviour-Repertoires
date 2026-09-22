# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import time
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from map_elites.map_elites_balance import MAPElites
from simulation.simulation_final import HexapodTorqueSimulation, evaluate_batch_parallel_torque
from controllers.hexapod_controller_final import RecurrentTorqueHexapodController
import config as experiment_config


from config import URDF_PATH
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bd", type=str, required=True)
    parser.add_argument("--fitness", type=str, default="combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--initial", type=int, default=5000)
    parser.add_argument("--mutation_rate", type=float, default=0.005)
    parser.add_argument("--mutation_strength", type=float, default=0.08)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # ==========================================================
    # PROJECT PATHS (synchronized with config.py)
    # ==========================================================

    # Prefer canonical paths defined in config.py (experiment_config) if available
    PROJECT_ROOT = getattr(experiment_config, 'PROJECT_ROOT', os.getcwd())
    RESULTS_DIR = getattr(experiment_config, 'RESULTS_DIR', os.path.join(PROJECT_ROOT, "results"))


    # ==========================================================
    # CONTROLLER
    # ==========================================================

    # Controller genome size - prefer value from config module
    CONTROLLER_GENOME_SIZE = getattr(experiment_config, 'CONTROLLER_GENOME_SIZE', RecurrentTorqueHexapodController.get_genome_size())

    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"URDF_PATH: {getattr(experiment_config, 'URDF_PATH', URDF_PATH)}")
    print(f"URDF exists: {os.path.exists(getattr(experiment_config, 'URDF_PATH', URDF_PATH))}")

    print(f"Genome size: {CONTROLLER_GENOME_SIZE}")

    # ==========================================================
    # BEHAVIOR DESCRIPTOR (from config.py) + local fitness configuration
    # ==========================================================

    BD_MODE = args.bd
    # Keep config.py as the source of truth, but sync its module globals to the notebook choice.
    experiment_config.BD_MODE = BD_MODE
    experiment_config.TRANSFORMER_BD_CONFIG = (
        experiment_config.get_transformer_bd_config(BD_MODE)
        if BD_MODE.startswith('transformer')
        else None
    )

    # Fitness configuration is intentionally local to the notebook.
    FITNESS_MODE = args.fitness
    DISTANCE_WEIGHT = 5.0 #3.0
    AVG_SPEED_WEIGHT = 0.8 #0.3
    STABILITY_PENALTY_WEIGHT = 0.05 #0.2
    ENERGY_WEIGHT = 2e-8 #2e-6

    def get_fitness_name(mode=None):
        if mode is None:
            mode = FITNESS_MODE

        if mode == 'combined':
            return 'Combined'
        elif mode == 'distance':
            return 'Distance_Traveled'
        elif mode == 'forward_velocity':
            return 'Forward_velocity'
        elif mode == 'avg_speed':
            return 'Average_speed'
        elif mode == 'stability':
            return 'Stability_score'
        else:
            raise ValueError(f"Unknown FITNESS_MODE '{mode}'")

    # Keep using the functions from config.py.
    get_behavior_descriptor_dims = experiment_config.get_behavior_descriptor_dims
    get_behavior_descriptor_names = experiment_config.get_behavior_descriptor_names
    TRANSFORMER_BD_CONFIG = experiment_config.TRANSFORMER_BD_CONFIG

    # ==========================================================
    # MAP-ELITES CONFIGURATION (local experiment override)
    # ==========================================================

    MAP_ELITES_CONFIG = {
        'behavior_descriptors_dims': get_behavior_descriptor_dims(),
        'genome_size': CONTROLLER_GENOME_SIZE,
        'genome_bounds': (-3, 3),
        'num_initial_random': args.initial,
        'num_iterations': args.iterations,
        'batch_size': 64,
        'mutation_rate': args.mutation_rate,
        'mutation_strength': args.mutation_strength,
        'bd_handling': 'clip' if BD_MODE.startswith('transformer') else 'reject',
        'num_repeated_evaluations': 3,
        'seed': 42,
    }
    print(behaviour_descriptors_dims := MAP_ELITES_CONFIG['behavior_descriptors_dims'])

    # ==========================================================
    # LOGGING CONFIGURATION (local experiment override)
    # ==========================================================

    LOGGING_CONFIG = {
        'save_interval': 500,
        'trajectory_save_interval': 500,
        'save_random_trajectories': False,
        'trajectory_save_probability': 0.1,
        'print_interval': 1000,
        'save_grid': True,
        'save_plots': True,
        'save_detailed_logs': False,
        'num_best_to_log': 20,
        'num_random_to_log': 20,
    }

    experiment_config.LOGGING_CONFIG = LOGGING_CONFIG

    # ==========================================================
    # SIMULATION CONFIGURATION (local experiment override)
    # ==========================================================

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


    timestamp = time.strftime("%Y%m%d_%H%M%S")

    experiment_name = (
        args.run_name
        if args.run_name is not None
        else f"{BD_MODE}_{FITNESS_MODE}_{timestamp}"
    )

    RESULTS_DIR = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    RANDOM_TRAJECTORIES_DIR = os.path.join(RESULTS_DIR, "random_trajectories")
    os.makedirs(RANDOM_TRAJECTORIES_DIR, exist_ok=True)
    DETAILED_LOGS_DIR = os.path.join(RESULTS_DIR, "detailed_logs")
    os.makedirs(DETAILED_LOGS_DIR, exist_ok=True)

    RESUME = False
    CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "map_elites_iter_4500.pkl"
    )

    start_iteration = 0

    if RESUME:
        print(CHECKPOINT)
        start_iteration = 4501

    # ==========================================================
    # DETAILED LOG EXPORT HELPERS
    # ==========================================================

    def save_detailed_logs_snapshot(map_elites, sim, iteration, num_best=20):
        cells = list(map_elites.grid.values())
        if not cells:
            print(f"No archive cells available to log at iteration {iteration}.")
            return None

        cells_sorted = sorted(cells, key=lambda x: x['fitness'], reverse=True)
        best_cells = cells_sorted[:num_best]

        detailed_logs_data = []
        for rank, cell in enumerate(best_cells, start=1):
            result = sim.evaluate_controller(
                cell['genome'],
                duration=SIMULATION_CONFIG['duration'],
                warmup_time=SIMULATION_CONFIG.get('warmup_time', 1.0),
                render=False,
                sample_interval=SIMULATION_CONFIG['sample_interval'],
                controller_interval=SIMULATION_CONFIG.get('controller_interval', 4),
                log_all_steps=True,
            )

            detailed_logs_data.append({
                'rank': rank,
                'fitness': float(cell['fitness']),
                # archive BD (from the cell) and run-computed BD (from evaluate_controller)
                'archive_behavior_descriptors': cell.get('bd'),
                'behavior_descriptors': result.get('behavior_descriptors', cell.get('bd')),
                'genome': cell['genome'],
                'detailed_log': result.get('detailed_log', []),
                'distance': float(result.get('distance', 0.0)),
                'energy': float(result.get('energy', 0.0)),
            })

        os.makedirs(DETAILED_LOGS_DIR, exist_ok=True)
        log_path = os.path.join(DETAILED_LOGS_DIR, f"detailed_logs_iter_{iteration}.pkl")
        with open(log_path, 'wb') as f:
            pickle.dump(detailed_logs_data, f)

        print(f"Saved detailed logs: {log_path} ({len(detailed_logs_data)} controllers)")
        return log_path

    # ==========================================================
    # CREATE MAP-ELITES
    # ==========================================================

    map_elites = MAPElites(
        behavior_descriptors_dims=MAP_ELITES_CONFIG['behavior_descriptors_dims'],
        genome_size=MAP_ELITES_CONFIG['genome_size'],
        genome_bounds=MAP_ELITES_CONFIG['genome_bounds'],
        bd_handling=MAP_ELITES_CONFIG['bd_handling'],
        seed=MAP_ELITES_CONFIG['seed'],
    )

    if RESUME and os.path.exists(CHECKPOINT):
        map_elites.load(CHECKPOINT)
        print(f"Resumed from {CHECKPOINT}")

    else:

        map_elites.bd_mode = BD_MODE
        map_elites.fitness_mode = FITNESS_MODE
        map_elites.bd_labels = get_behavior_descriptor_names()
        map_elites.fitness_label = get_fitness_name()

        # ==========================================================
        # INITIAL RANDOM POPULATION
        # ==========================================================

        initial_population = []

        for _ in range(args.initial):
            initial_population.append(map_elites.random_genome())

        # ==========================================================
        # EVALUATE INITIAL POPULATION
        # ==========================================================

        fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties = evaluate_batch_parallel_torque(initial_population)

        # ==========================================================
        # ADD TO ARCHIVE
        # ==========================================================

        for genome, fitness, bd, final_position, distance, average_speed, energy, stability_penalty in zip(
            initial_population,
            fitnesses,
            bds,
            final_positions,
            distances,
            average_speeds,
            energies,
            stability_penalties,
        ):
            map_elites.add_to_grid(
                genome,
                fitness,
                bd,
                final_position=final_position,
                distance=distance,
                average_speed=average_speed,
                energy=energy,
                stability_penalty=stability_penalty,
            )

        # ==========================================================
        # INITIAL STATS
        # ==========================================================

        stats = map_elites.get_statistics()

        print("\nInitial archive statistics:\n")

        for k, v in stats.items():
            print(f"{k}: {v}")

        # ==========================================================
        # TIMER
        # ==========================================================

        start_time = time.time()

    bds = np.asarray(bds)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        bds[:, 0],
        bds[:, 1],
        s=4,
        alpha=0.4
    )

    plt.xlabel("Final X position (m)")
    plt.ylabel("Final Y position (m)")
    plt.title("Distribution of random controllers")

    # Show archive limits
    x_low, x_high, _ = MAP_ELITES_CONFIG['behavior_descriptors_dims'][0]
    y_low, y_high, _ = MAP_ELITES_CONFIG['behavior_descriptors_dims'][1]

    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)

    plt.grid(True)
    plt.gca().set_aspect('equal')

    plt.savefig(
        os.path.join(RESULTS_DIR, "random_population.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    visited_cells = set()

    for bd in bds:
        norm_bd = map_elites.process_behavior_descriptors(bd)

        if norm_bd is None:
            continue

        idx = map_elites.get_grid_index(norm_bd)

        if idx is not None:
            visited_cells.add(tuple(idx))

    print(f"Unique cells visited: {len(visited_cells)}")
    print(f"Archive cells: {map_elites.get_total_cells()}")
    print(f"Coverage from random controllers: "
        f"{100*len(visited_cells)/map_elites.get_total_cells():.2f}%")

    # ==========================================================
    # CELL 2 — MAIN EVOLUTION LOOP
    # ==========================================================

    sim_for_logging = HexapodTorqueSimulation(
        urdf_path=URDF_PATH,
        gui=False,
        time_step=SIMULATION_CONFIG['time_step'],
    )

    for iteration in range(start_iteration, args.iterations):

        if iteration % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Iteration {iteration}")

        # ------------------------------------------------------
        # SAMPLE PARENTS
        # ------------------------------------------------------

        parents = []

        for i in range(MAP_ELITES_CONFIG['batch_size']):
            parents.append(map_elites.random_selection())

        # ------------------------------------------------------
        # MUTATE
        # ------------------------------------------------------

        offspring = []

        for parent in parents:

            child = map_elites.mutate(
                genome=parent,
                mutation_rate=MAP_ELITES_CONFIG['mutation_rate'],
                mutation_strength=MAP_ELITES_CONFIG['mutation_strength']
            )

            offspring.append(child)

        # ------------------------------------------------------
        # EVALUATE OFFSPRING
        # ------------------------------------------------------

        fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties = evaluate_batch_parallel_torque(offspring)

        # ------------------------------------------------------
        # ADD TO ARCHIVE
        # ------------------------------------------------------

        for genome, fitness, bd, final_position, distance, average_speed, energy, stability_penalty in zip(
            offspring,
            fitnesses,
            bds,
            final_positions,
            distances,
            average_speeds,
            energies,
            stability_penalties,
        ):
            map_elites.add_to_grid(
                genome,
                fitness,
                bd,
                final_position=final_position,
                distance=distance,
                average_speed=average_speed,
                energy=energy,
                stability_penalty=stability_penalty,
            )

        # ------------------------------------------------------
        # UPDATE HISTORY
        # ------------------------------------------------------

        map_elites.update_history()

        # ------------------------------------------------------
        # CHECKPOINT
        # ------------------------------------------------------

        if iteration % LOGGING_CONFIG['save_interval'] == 0:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"map_elites_iter_{iteration}.pkl"
            )

            map_elites.save(checkpoint_path)

            if LOGGING_CONFIG.get('save_detailed_logs', False):
                save_detailed_logs_snapshot(
                    map_elites,
                    sim_for_logging,
                    iteration=iteration,
                    num_best=LOGGING_CONFIG.get('num_best_to_log', 20),
                )

            stats = map_elites.get_statistics()

            elapsed = time.time() - start_time

            print("\n================================================")
            print(f"Iteration: {iteration}")
            print("================================================")
            print(f"Coverage:      {stats['coverage']:.4f}")
            print(f"Filled cells:  {stats['num_filled']}")
            print(f"Max fitness:   {stats['max_fitness']:.4f}")
            print(f"QD score:      {stats['qd_score']:.4f}")
            print(f"Elapsed:       {elapsed/60:.2f} min")
            print("================================================\n")

    # ==========================================================
    # SAVE FINAL ARCHIVE
    # ==========================================================

    final_archive_path = os.path.join(RESULTS_DIR, "map_elites_final.pkl")
    map_elites.save(final_archive_path)
    print(f"Final archive saved to: {final_archive_path}")

    if LOGGING_CONFIG.get('save_detailed_logs', False):
        save_detailed_logs_snapshot(
            map_elites,
            sim_for_logging,
            iteration='final',
            num_best=LOGGING_CONFIG.get('num_best_to_log', 20),
        )

    final_stats = map_elites.get_statistics()
    print("\nFinal archive statistics:\n")
    for k, v in final_stats.items():
        print(f"{k}: {v}")

    sim_for_logging.disconnect()

    # ==========================================================
    # VISUALIZE MAP-ELITES GRID FROM SAVED ARCHIVE
    # ==========================================================

    # Prefer final archive, fallback to latest checkpoint
    final_archive_path = os.path.join(RESULTS_DIR, "map_elites_final.pkl")
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith('map_elites_iter_') and f.endswith('.pkl')
    ]

    archive_path = None
    if os.path.exists(final_archive_path):
        archive_path = final_archive_path
    elif checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
        archive_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])

    if archive_path is not None:
        print(f"Loading archive: {archive_path}")
        with open(archive_path, 'rb') as f:
            loaded_data = pickle.load(f)

        archive = loaded_data

        print("=" * 60)
        print("MAP-ELITES GRID VISUALIZATION")
        print("=" * 60)

        # Get statistics
        if isinstance(archive, dict) and 'grid' in archive:
            grid = archive.get('grid', {})
            bd_labels = archive.get('bd_labels', ['Dimension 1', 'Dimension 2'])
            fitness_label = archive.get('fitness_label', 'Fitness')
            grid_shape = archive.get('grid_shape', None)

            all_fitnesses = [cell['fitness'] for cell in grid.values()]
            median_fitness = np.median(all_fitnesses) if all_fitnesses else 0.0
            qd_score = sum(all_fitnesses)
            num_filled = len(grid)
            coverage = num_filled / np.prod(grid_shape) if grid_shape else 0.0
            max_fitness = max(all_fitnesses) if all_fitnesses else 0.0
            mean_fitness = np.mean(all_fitnesses) if all_fitnesses else 0.0
        else:
            stats = archive.get_statistics()
            all_fitnesses = [cell['fitness'] for cell in archive.grid.values()]
            median_fitness = np.median(all_fitnesses) if all_fitnesses else 0.0
            qd_score = sum(all_fitnesses)
            num_filled = stats['num_filled']
            coverage = stats['coverage']
            max_fitness = stats['max_fitness']
            mean_fitness = stats['mean_fitness']
            bd_labels = archive.bd_labels
            fitness_label = archive.fitness_label
            grid_shape = archive.grid_shape
            grid = archive.grid

        print(f"Coverage:      {coverage*100:.2f}%")
        print(f"Filled cells:  {num_filled}")
        print(f"Total cells:   {np.prod(grid_shape)}")
        print(f"QD-score:      {qd_score:.4f}")
        print(f"Max fitness:   {max_fitness:.4f}")
        print(f"Mean fitness:  {mean_fitness:.4f}")
        print(f"Median fitness: {median_fitness:.4f}")

        # Create heatmap for 2D behavior descriptors
        if len(grid_shape) == 2:
            heatmap = np.full(grid_shape, np.nan)  # Initialize with NaN (empty cells)
            for grid_idx, cell in grid.items():
                heatmap[grid_idx] = cell['fitness']

            plt.figure(figsize=(10, 8))
            im = plt.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(im, label='Fitness')
            plt.xlabel(bd_labels[0])
            plt.ylabel(bd_labels[1])
            plt.title(f"MAP-Elites Grid: {fitness_label}\n(white = empty cells, colored = solutions)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(RESULTS_DIR, "final_grid.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            print("\nGrid heatmap plotted (white areas = no solutions).")
        else:
            print(f"\n(Grid is {len(grid_shape)}D; heatmap skipped)")
    else:
        print(f"No final archive or checkpoints found in {RESULTS_DIR}")

    summary = {
        "bd_mode": BD_MODE,
        "fitness_mode": FITNESS_MODE,
        "seed": args.seed,
        "coverage": final_stats["coverage"],
        "filled_cells": final_stats["num_filled"],
        "max_fitness": final_stats["max_fitness"],
        "qd_score": final_stats["qd_score"],
    }

    with open(
        os.path.join(RESULTS_DIR, "summary.json"),
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()