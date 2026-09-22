import os
import time
import pickle
import json
import numpy as np
import torch

from map_elites.map_elites import MAPElites
from simulation.simulation_goal import (
    HexapodTorqueSimulation,
    evaluate_batch_parallel_goal_conditioned
)
from controllers.hexapod_goal_controller import GoalDirectedRecurrentTorqueHexapodController
from config import MAP_ELITES_CONFIG, URDF_PATH
import config as experiment_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# SIMPLE LOGGER (replaces all prints)
# ==========================================================
from datetime import datetime

class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")


def main():

    # ==========================================================
    # OUTPUT LOG FILE
    # ==========================================================
    log_path = os.path.join(
        experiment_config.RESULTS_DIR
        if hasattr(experiment_config, "RESULTS_DIR")
        else "results",
        "run_log.txt"
    )
    logger = Logger(log_path)

    start_time = time.time()

    # ==========================================================
    # PATHS
    # ==========================================================
    PROJECT_ROOT = getattr(experiment_config, 'PROJECT_ROOT', os.getcwd())
    RESULTS_DIR = getattr(experiment_config, 'RESULTS_DIR', os.path.join(PROJECT_ROOT, "results"))

    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ==========================================================
    # CONFIG
    # ==========================================================
    CONTROLLER_GENOME_SIZE = getattr(
        experiment_config,
        'CONTROLLER_GENOME_SIZE',
        GoalDirectedRecurrentTorqueHexapodController.get_genome_size()
    )

    BD_MODE = "duty_factor"  # options: 'xy_position', 'transformer_latent', 'transformer_8d', 'duty_factor'
    FITNESS_MODE = "combined"

    experiment_config.BD_MODE = BD_MODE
    experiment_config.TRANSFORMER_BD_CONFIG = (
        experiment_config.get_transformer_bd_config(BD_MODE)
        if BD_MODE.startswith("transformer")
        else None
    )

    MAP_ELITES_CONFIG = {
        'behavior_descriptors_dims': experiment_config.get_behavior_descriptor_dims(),
        'genome_size': CONTROLLER_GENOME_SIZE,
        'genome_bounds': (-0.75, 0.75),
        'num_initial_random': 50000,
        'num_iterations': 10000,
        'batch_size': 64,
        'mutation_rate': 0.2,
        'mutation_strength': 0.15,
        'bd_handling': 'clip',
        'seed': 42,
    }

    LOGGING_CONFIG = {
        'save_interval': 500,
        'progress_interval': 100,
    }

    SIMULATION_CONFIG = {
        'duration': 6.0,
        'time_step': 1.0 / 240,
        'sample_interval': 1.0,
        'controller_interval': 4,
        'warmup_time': 1.0,
    }

    RESUME = False
    CHECKPOINT = os.path.join(
        RESULTS_DIR,
        "checkpoints",
        "map_elites_8000.pkl"
    )

    start_iteration = 0

    if RESUME:

        start_iteration = 8001

    logger.log("=" * 80)
    logger.log("EXPERIMENT START")
    logger.log(f"Project root: {PROJECT_ROOT}")
    logger.log(f"URDF: {URDF_PATH}")
    logger.log(f"Device: {DEVICE}")
    logger.log(f"BD_MODE: {BD_MODE}")
    logger.log(f"FITNESS_MODE: {FITNESS_MODE}")
    logger.log(f"Genome size: {CONTROLLER_GENOME_SIZE}")
    logger.log(f"Initial random population: {MAP_ELITES_CONFIG['num_initial_random']}")
    logger.log(f"Iterations: {MAP_ELITES_CONFIG['num_iterations']}")
    logger.log(f"Batch size: {MAP_ELITES_CONFIG['batch_size']}")
    logger.log(f"Mutation rate: {MAP_ELITES_CONFIG['mutation_rate']}")
    logger.log(f"Mutation strength: {MAP_ELITES_CONFIG['mutation_strength']}")
    logger.log("=" * 80)

    # ==========================================================
    # MAP-ELITES INIT
    # ==========================================================

    map_elites = MAPElites(
        behavior_descriptors_dims=MAP_ELITES_CONFIG['behavior_descriptors_dims'],
        genome_size=MAP_ELITES_CONFIG['genome_size'],
        genome_bounds=MAP_ELITES_CONFIG['genome_bounds'],
        bd_handling=MAP_ELITES_CONFIG['bd_handling'],
        seed=MAP_ELITES_CONFIG['seed'],
    )

    #print("MAP-Elites BD dims:")
    #print(map_elites.bd_dims)
    #print("num dims:", len(map_elites.bd_dims))

    if RESUME and os.path.exists(CHECKPOINT):

        map_elites.load(CHECKPOINT)

        logger.log(f"Resumed from {CHECKPOINT}")

    else:

        map_elites.bd_mode = BD_MODE


    # ==========================================================
    # INITIAL POPULATION
    # ==========================================================
    if not (RESUME and os.path.exists(CHECKPOINT)):

        initial_population = [
            map_elites.random_genome()
            for _ in range(MAP_ELITES_CONFIG['num_initial_random'])
        ]

        fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties = (
            evaluate_batch_parallel_goal_conditioned(
                initial_population
            )
        )

        for g, f, bd, fp, distance, average_speed, energy, stability_penalty in zip(
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
                g,
                f,
                bd,
                final_position=fp,
                distance=distance,
                average_speed=average_speed,
                energy=energy,
                stability_penalty=stability_penalty,
            )

    stats = map_elites.get_statistics()

    logger.log(
        f"[INITIAL] "
        f"coverage={stats['coverage']:.4f} "
        f"filled={stats['num_filled']} "
        f"max_fit={stats['max_fitness']:.4f} "
        f"qd={stats['qd_score']:.4f}"
    )

    #print BDs
    #for i, bd in enumerate(bds):
    #    print(f"Genome {i}: BD = {bd}, fitness = {fitnesses[i]}")


    loop_start_time = time.time()
    # ==========================================================
    # EVOLUTION LOOP
    # ==========================================================
    sim_for_logging = HexapodTorqueSimulation(
        urdf_path=URDF_PATH,
        gui=False,
        time_step=SIMULATION_CONFIG['time_step'],
    )

    for iteration in range(start_iteration, MAP_ELITES_CONFIG['num_iterations']):

        parents = [
            map_elites.random_selection()
            for _ in range(MAP_ELITES_CONFIG['batch_size'])
        ]

        offspring = [
            map_elites.mutate(
                p,
                MAP_ELITES_CONFIG['mutation_rate'],
                MAP_ELITES_CONFIG['mutation_strength']
            )
            for p in parents
        ]

        fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties = evaluate_batch_parallel_goal_conditioned(offspring)

        for g, f, bd, fp, distance, average_speed, energy, stability_penalty in zip(offspring, fitnesses, bds, final_positions, distances, average_speeds, energies, stability_penalties):
            map_elites.add_to_grid(g, f, bd, final_position=fp, distance=distance, average_speed=average_speed, energy=energy, stability_penalty=stability_penalty)

        map_elites.update_history()

        if (
            iteration == 0
            or (iteration + 1) % LOGGING_CONFIG['progress_interval'] == 0
            or iteration == MAP_ELITES_CONFIG['num_iterations'] - 1
        ):

            completed = iteration + 1
            total = MAP_ELITES_CONFIG['num_iterations']

            elapsed = time.time() - loop_start_time

            seconds_per_iteration = elapsed / completed

            remaining_iterations = total - completed
            eta_seconds = remaining_iterations * seconds_per_iteration

            stats = map_elites.get_statistics()

            logger.log(
                f"[PROGRESS] "
                f"{completed}/{total} "
                f"({100.0 * completed / total:.1f}%) | "
                f"coverage={stats['coverage']:.4f} | "
                f"filled={stats['num_filled']} | "
                f"max_fit={stats['max_fitness']:.4f} | "
                f"qd={stats['qd_score']:.4f} | "
                f"{seconds_per_iteration:.2f}s/iter | "
                f"ETA={eta_seconds/60:.1f} min"
            )

        # checkpoint
        if iteration % LOGGING_CONFIG['save_interval'] == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"map_elites_{iteration}.pkl")
            map_elites.save(ckpt_path)

            stats = map_elites.get_statistics()
            elapsed = time.time() - start_time

            logger.log(
                f"[CHECKPOINT] "
                f"iteration={iteration} "
                f"path={ckpt_path} "
                f"coverage={stats['coverage']:.4f} "
                f"filled={stats['num_filled']} "
                f"max_fit={stats['max_fitness']:.4f}"
            )

    # ==========================================================
    # FINAL SAVE
    # ==========================================================
    final_path = os.path.join(RESULTS_DIR, "map_elites_final.pkl")
    map_elites.save(final_path)

    sim_for_logging.disconnect()

    final_stats = map_elites.get_statistics()
    total_runtime = time.time() - start_time

    logger.log("=" * 80)
    logger.log("EXPERIMENT FINISHED")
    logger.log(f"Runtime: {total_runtime/60:.2f} minutes")
    logger.log(f"Runtime: {total_runtime/3600:.2f} hours")
    logger.log(f"Final stats: {final_stats}")
    logger.log("=" * 80)

    # ==========================================================
    # EXPORT TOP SOLUTIONS
    # ==========================================================
    grid = map_elites.grid

    top = sorted(grid.items(), key=lambda x: x[1]['fitness'], reverse=True)[:5]

    export = {
        "solutions": []
    }

    for i, (idx, cell) in enumerate(top):
        export["solutions"].append({
            "rank": i + 1,
            "fitness": float(cell["fitness"]),
            "genome": (
                cell["genome"].tolist()
                if hasattr(cell["genome"], "tolist")
                else cell["genome"]
            ),
        })

    out_json = os.path.join(RESULTS_DIR, "top_solutions.json")
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)

    logger.log(f"Exported top solutions: {out_json}")
    logger.log("Experiment finished")


if __name__ == "__main__":
    main()
