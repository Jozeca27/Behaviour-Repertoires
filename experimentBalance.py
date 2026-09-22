def main():
    # ==========================================================
    # IMPORTS
    # ==========================================================

    import os
    import importlib
    import time
    import pickle
    import numpy as np
    from map_elites.map_elites import MAPElites
    from simulation.simulation_final import HexapodTorqueSimulation, evaluate_batch_parallel_torque
    from controllers.hexapod_controller_final import RecurrentTorqueHexapodController
    from config import URDF_PATH
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import config as experiment_config

    import logging
    import sys

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


    # ==========================================================
    # BEHAVIOR DESCRIPTOR (from config.py) + local fitness configuration
    # ==========================================================

    BD_MODE = 'xy_position'  # Options: 'xy_position', 'duty_factor', 'transformer_duty_factor', 'transformer_duty_factor_with_goal'

    # Keep config.py as the source of truth, but sync its module globals to the notebook choice.
    experiment_config.BD_MODE = BD_MODE
    experiment_config.TRANSFORMER_BD_CONFIG = (
        experiment_config.get_transformer_bd_config(BD_MODE)
        if BD_MODE.startswith('transformer')
        else None
    )

    # Fitness configuration is intentionally local to the notebook.
    FITNESS_MODE = 'distance'  # Options: 'combined', 'distance', 'forward_velocity', 'avg_speed', 'stability'
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


    LOG_FILE = os.path.join(RESULTS_DIR, "run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)


    logger.info("=" * 60)
    logger.info("PROJECT CONFIGURATION")
    logger.info("=" * 60)

    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"URDF_PATH: {getattr(experiment_config, 'URDF_PATH', URDF_PATH)}")
    logger.info(f"URDF exists: {os.path.exists(getattr(experiment_config, 'URDF_PATH', URDF_PATH))}")

    logger.info(f"Genome size: {CONTROLLER_GENOME_SIZE}")

    logger.info(f"Behavior Descriptor Mode: {BD_MODE}")

    # ==========================================================
    # MAP-ELITES CONFIGURATION (local experiment override)
    # ==========================================================

    MAP_ELITES_CONFIG = {
        'behavior_descriptors_dims': get_behavior_descriptor_dims(),
        'genome_size': CONTROLLER_GENOME_SIZE,
        'genome_bounds': (-3, 3),
        'num_initial_random': 5000,
        'num_iterations': 5000,
        'batch_size': 64,
        'mutation_rate': 0.005,
        'mutation_strength': 0.08,
        'bd_handling': 'clip' if BD_MODE.startswith('transformer') else 'reject',
        'num_repeated_evaluations': 3,
        'seed': 42,
    }
    logger.info(behaviour_descriptors_dims := MAP_ELITES_CONFIG['behavior_descriptors_dims'])

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

    experiment_name = f"map_elites_{BD_MODE}_{get_fitness_name()}_fitness"
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
        logger.info(CHECKPOINT)
        start_iteration = 4501

    # ==========================================================
    # DETAILED LOG EXPORT HELPERS
    # ==========================================================

    def save_detailed_logs_snapshot(map_elites, sim, iteration, num_best=20):
        cells = list(map_elites.grid.values())
        if not cells:
            logger.info(f"No archive cells available to log at iteration {iteration}.")
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

        logger.info(f"Saved detailed logs: {log_path} ({len(detailed_logs_data)} controllers)")
        return log_path

    # ==========================================================
    # CREATE MAP-ELITES No smoothing
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
        logger.info(f"Resumed from {CHECKPOINT}")

    else:

        map_elites.bd_mode = BD_MODE
        map_elites.fitness_mode = FITNESS_MODE
        map_elites.bd_labels = get_behavior_descriptor_names()
        map_elites.fitness_label = get_fitness_name()

        # ==========================================================
        # INITIAL RANDOM POPULATION
        # ==========================================================

        initial_population = []

        for i in range(MAP_ELITES_CONFIG['num_initial_random']):
            initial_population.append(map_elites.random_genome())

        # ==========================================================
        # EVALUATE INITIAL POPULATION
        # ==========================================================

        fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties = evaluate_batch_parallel_torque(initial_population)

        # ==========================================================
        # ADD TO ARCHIVE
        # ==========================================================

        for i, (genome, fitness, bd, final_position, distance, average_speed, energy, stability_penalty) in enumerate(zip(initial_population, fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties), start=1):
            map_elites.add_to_grid(genome, fitness, bd, final_position=final_position, distance=distance, average_speed=average_speed, energy=energy, stability_penalty=stability_penalty)

        # ==========================================================
        # INITIAL STATS
        # ==========================================================

        stats = map_elites.get_statistics()

        logger.info("\nInitial archive statistics:\n")

        for k, v in stats.items():
            logger.info(f"{k}: {v}")

        # ==========================================================
        # TIMER
        # ==========================================================

        start_time = time.time()

        # ==========================================================
    # CELL 2 — MAIN EVOLUTION LOOP
    # ==========================================================

    sim_for_logging = HexapodTorqueSimulation(
        urdf_path=URDF_PATH,
        gui=False,
        time_step=SIMULATION_CONFIG['time_step'],
    )

    for iteration in range(
        start_iteration,
        MAP_ELITES_CONFIG['num_iterations']
    ):

        # ------------------------------------------------------
        # SAMPLE PARENTS
        # ------------------------------------------------------

        parents = []

        for iteration in range(
            start_iteration,
            MAP_ELITES_CONFIG['num_iterations']
        ):
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

        for genome, fitness, bd, final_position, distance, average_speed, energy, stability_penalty in zip(offspring, fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties):
            map_elites.add_to_grid(genome, fitness, bd, final_position=final_position, distance=distance, average_speed=average_speed, energy=energy, stability_penalty=stability_penalty)

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

            logger.info("\n================================================")
            logger.info(f"Iteration: {iteration}")
            logger.info("================================================")
            logger.info(f"Coverage:      {stats['coverage']:.4f}")
            logger.info(f"Filled cells:  {stats['num_filled']}")
            logger.info(f"Max fitness:   {stats['max_fitness']:.4f}")
            logger.info(f"QD score:      {stats['qd_score']:.4f}")
            logger.info(f"Elapsed:       {elapsed/60:.2f} min")
            logger.info("================================================\n")

    # ==========================================================
    # SAVE FINAL ARCHIVE
    # ==========================================================

    final_archive_path = os.path.join(RESULTS_DIR, "map_elites_final.pkl")
    map_elites.save(final_archive_path)
    logger.info(f"Final archive saved to: {final_archive_path}")

    if LOGGING_CONFIG.get('save_detailed_logs', False):
        save_detailed_logs_snapshot(
            map_elites,
            sim_for_logging,
            iteration='final',
            num_best=LOGGING_CONFIG.get('num_best_to_log', 20),
        )

    final_stats = map_elites.get_statistics()
    logger.info("\nFinal archive statistics:\n")
    for k, v in final_stats.items():
        logger.info(f"{k}: {v}")

    sim_for_logging.disconnect()

    # ==========================================================
    # VISUALIZE MAP-ELITES GRID FROM SAVED ARCHIVE
    # ==========================================================

    import matplotlib.pyplot as plt
    import os

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
        logger.info(f"Loading archive: {archive_path}")
        with open(archive_path, 'rb') as f:
            loaded_data = pickle.load(f)

        archive = loaded_data

        logger.info("=" * 60)
        logger.info("MAP-ELITES GRID VISUALIZATION")
        logger.info("=" * 60)

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

        logger.info(f"Coverage:      {coverage*100:.2f}%")
        logger.info(f"Filled cells:  {num_filled}")
        logger.info(f"Total cells:   {np.prod(grid_shape)}")
        logger.info(f"QD-score:      {qd_score:.4f}")
        logger.info(f"Max fitness:   {max_fitness:.4f}")
        logger.info(f"Mean fitness:  {mean_fitness:.4f}")
        logger.info(f"Median fitness: {median_fitness:.4f}")

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
            plt.show()

            logger.info("\nGrid heatmap plotted (white areas = no solutions).")
        else:
            logger.info(f"\n(Grid is {len(grid_shape)}D; heatmap skipped)")
    else:
        logger.info(f"No final archive or checkpoints found in {RESULTS_DIR}")

    # ==========================================================
    # EXPORT TOP SOLUTIONS TO STANDALONE VISUALIZATION SCRIPT
    # ==========================================================

    import json

    if 'archive' in globals():
        num_to_visualize = 5  # Export top 5 solutions

        # Extract grid from archive
        if isinstance(archive, dict) and 'grid' in archive:
            grid = archive.get('grid', {})
        else:
            grid = archive.grid

        # Sort cells by fitness (descending)
        sorted_cells = sorted(
            grid.items(),
            key=lambda x: x[1]['fitness'],
            reverse=True
        )

        top_cells = sorted_cells[:num_to_visualize]

        # Prepare data for export
        solutions_data = {
            'solutions': [],
            'config': {
                'urdf_path': URDF_PATH,
                'duration': SIMULATION_CONFIG['duration'],
                'time_step': SIMULATION_CONFIG['time_step'],
                'sample_interval': SIMULATION_CONFIG['sample_interval'],
            }
        }

        logger.info(f"Exporting top {len(top_cells)} solutions...")
        logger.info("=" * 60)

        for rank, (grid_idx, cell) in enumerate(top_cells, 1):
            genome = cell['genome']
            fitness = cell['fitness']
            bd = cell['bd']

            logger.info(f"[{rank}] Fitness: {fitness:.4f}, BD: {bd}, Grid: {grid_idx}")

            solutions_data['solutions'].append({
                'rank': rank,
                'grid_idx': grid_idx,
                'fitness': float(fitness),
                'bd': bd.tolist() if hasattr(bd, 'tolist') else bd,
                'genome': genome.tolist() if hasattr(genome, 'tolist') else genome,
            })

        # Save to JSON file
        export_path = os.path.join(RESULTS_DIR, 'top_solutions.json')
        with open(export_path, 'w') as f:
            json.dump(solutions_data, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info(f"Solutions exported to: {export_path}")
        logger.info("\nTo visualize, run:")
        logger.info("  ./bulletenv/bin/python visualize_top_solutions.py")
    else:
        logger.info("No archive loaded. Run the previous visualization/load cell first.")


if __name__ == "__main__":
    main()
