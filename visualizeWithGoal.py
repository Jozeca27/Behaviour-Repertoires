#!/usr/bin/env python3

import json
import os
import time
import numpy as np

import config
from config import URDF_PATH
from simulation.simulation_goal import HexapodTorqueSimulation


def main():

    #results_file = os.path.join( config.RESULTS_DIR, "top_solutions.json")
    results_file = '/home/joze/Mestrado/Dissertacao/Pybullet/SaveResults/Goal/GoalTrans/top_solutions.json'

    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Could not find: {results_file}"
        )

    with open(results_file, "r") as f:
        data = json.load(f)

    solutions = data["solutions"]

    print(f"\nLoaded {len(solutions)} solutions")

    sim = HexapodTorqueSimulation(
        urdf_path=URDF_PATH,
        gui=True,
        time_step=1.0 / 240,
    )

    sim.connect()

    try:

        for solution in solutions:

            rank = solution["rank"]
            fitness = solution["fitness"]
            genome = np.asarray(
                solution["genome"],
                dtype=np.float32
            )

            print("\n" + "=" * 80)
            print(f"RANK {rank}")
            print(f"Stored fitness: {fitness:.4f}")
            print("=" * 80)

            for goal_idx, goal in enumerate(config.GOAL_VECTORS):

                print(
                    f"\nGoal {goal_idx + 1}/{len(config.GOAL_VECTORS)}"
                )
                print(f"Goal vector: {goal}")

                result = sim.evaluate_controller(
                    genome,
                    duration=6.0,
                    warmup_time=1.0,
                    render=True,
                    controller_interval=4,
                    goal_vector=goal,
                )

                print(
                    f"Fitness: "
                    f"{result['fitness']:.4f}"
                )

                print(
                    f"Distance: "
                    f"{result['distance']:.4f}"
                )

                print(
                    f"Energy: "
                    f"{result['energy']:.4f}"
                )

                if result["final_position"] is not None:

                    fp = result["final_position"]

                    print(
                        f"Final position: "
                        f"({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f})"
                    )

                time.sleep(2)

            input(
                "\nPress ENTER to view the next solution..."
            )

    finally:

        sim.disconnect()


if __name__ == "__main__":
    main()