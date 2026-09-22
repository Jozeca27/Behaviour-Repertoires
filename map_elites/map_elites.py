"""
MAP-Elites Algorithm Implementation
Quality Diversity algorithm with configurable BD handling,
improved mutation, reproducibility, noisy evaluation support,
and float32 genome storage.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import pickle

from controllers.hexapod_controller_final import RecurrentTorqueHexapodController


class MAPElites:

    def __init__(
        self,
        behavior_descriptors_dims: List[Tuple[float, float, int]],
        genome_size: int,
        genome_bounds: Tuple[float, float] = (-1.0, 1.0),

        # NEW: configurable BD handling
        bd_handling: str = "reject",  # "reject" or "clip"

        # NEW: reproducibility
        seed: Optional[int] = None
    ):

        # ------------------------------
        # Reproducibility
        # ------------------------------
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # ------------------------------
        # Core config
        # ------------------------------
        self.bd_dims = behavior_descriptors_dims
        self.genome_size = genome_size
        self.genome_bounds = genome_bounds
        self.bd_handling = bd_handling.lower()

        if self.bd_handling not in ["reject", "clip"]:
            raise ValueError(
                "bd_handling must be either 'reject' or 'clip'"
            )

        # ------------------------------
        # Metadata
        # ------------------------------
        self.bd_mode = None
        self.fitness_mode = None
        self.bd_labels = None
        self.fitness_label = None

        # ------------------------------
        # Grid definition
        # ------------------------------
        self.grid_shape = tuple(dim[2] for dim in behavior_descriptors_dims)
        self.grid = {}

        # ------------------------------
        # Stats
        # ------------------------------
        self.num_evaluations = 0
        self.invalid_bd_count = 0

        self.history = {
            'coverage': [],
            'max_fitness': [],
            'mean_fitness': [],
            'qd_score': [],
            'evaluations': []
        }

    # ==========================================================
    # BEHAVIOR DESCRIPTOR PROCESSING
    # ==========================================================

    def process_behavior_descriptors(
        self,
        behavior_descriptors: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Process behavior descriptors according to configured policy.

        Policies:
            - reject:
                reject descriptors outside bounds

            - clip:
                clip descriptors to bounds
        """

        if len(behavior_descriptors) != len(self.bd_dims):
            raise ValueError("Behavior descriptor dimension mismatch")

        processed = []

        for val, (low, high, _) in zip(
            behavior_descriptors,
            self.bd_dims
        ):

            # -----------------------------------
            # Reject mode
            # -----------------------------------
            if self.bd_handling == "reject":

                if val < low or val > high:
                    self.invalid_bd_count += 1
                    return None

            # -----------------------------------
            # Clip mode
            # -----------------------------------
            elif self.bd_handling == "clip":

                val = np.clip(val, low, high)

            # -----------------------------------
            # Normalize to [0,1]
            # -----------------------------------
            normalized = (val - low) / (high - low)

            processed.append(normalized)

        return np.array(processed, dtype=np.float32)

    def get_grid_index(
        self,
        normalized_bd: np.ndarray
    ) -> Optional[Tuple[int, ...]]:
        """
        Convert normalized BD [0,1] into grid index.
        """

        indices = []

        for val, (_, _, bins) in zip(normalized_bd, self.bd_dims):

            if val < 0.0 or val > 1.0:
                return None

            idx = int(val * bins)

            # Handle upper edge exactly at 1.0
            idx = min(idx, bins - 1)

            indices.append(idx)

        return tuple(indices)

    # ==========================================================
    # CORE ARCHIVE FUNCTION
    # ==========================================================

    def add_to_grid(
        self,
        genome: np.ndarray,
        fitness: float,
        behavior_descriptors: np.ndarray,
        fitness_std: float = 0.0,
        final_position: Optional[np.ndarray] = None,
        distance: float = None,
        average_speed: float = None,
        energy: float = None,
        stability_penalty: float = None,
        
    ) -> bool:
        """
        Add solution if it improves the elite in a cell.
        """

        norm_bd = self.process_behavior_descriptors(
            behavior_descriptors
        )

        if norm_bd is None:
            return False

        grid_idx = self.get_grid_index(norm_bd)

        if grid_idx is None:
            self.invalid_bd_count += 1
            return False

        # ------------------------------------------------------
        # Replace elite if better
        # ------------------------------------------------------
        if (
            grid_idx not in self.grid
            or self.grid[grid_idx]['fitness'] < fitness
        ):

            self.grid[grid_idx] = {

                # Store as float32 to reduce memory usage
                'genome': genome.astype(np.float32),
                'fitness': float(fitness),
                'fitness_std': float(fitness_std),

                "distance": None if distance is None else float(distance),
                "average_speed": None if average_speed is None else float(average_speed),
                "energy": None if energy is None else float(energy),
                "stability_penalty": None if stability_penalty is None else float(stability_penalty),

                'bd': norm_bd.astype(np.float32),
                # Optional end-of-episode base position for post-hoc selection
                'final_position': (
                    None if final_position is None
                    else np.array(final_position, dtype=np.float32)
                )
            }

            return True

        return False

    # ==========================================================
    # SAMPLING
    # ==========================================================

    def random_genome(self) -> np.ndarray:
            return self.rng.uniform(
                self.genome_bounds[0],
                self.genome_bounds[1],
                self.genome_size
            ).astype(np.float32)

    def random_selection(self) -> np.ndarray:
        """
        Uniformly sample an elite from the archive.
        """

        if not self.grid:
            return self.random_genome()

        keys = list(self.grid.keys())

        selected_key = keys[self.rng.integers(len(keys))]

        return self.grid[selected_key]['genome'].copy()

    # ==========================================================
    # IMPROVED MUTATION
    # ==========================================================

    def mutate(
        self,
        genome: np.ndarray,
        mutation_rate: float,
        mutation_strength: float
    ) -> np.ndarray:
        """
        Bernoulli per-gene mutation.

        Each gene has probability mutation_rate
        of being perturbed by Gaussian noise.
        """

        child = genome.copy()

        # ------------------------------------------------------
        # Bernoulli mutation mask
        # ------------------------------------------------------
        mutation_mask = (
            self.rng.random(self.genome_size)
            < mutation_rate
        )

        num_mutations = np.sum(mutation_mask)

        # Guarantee at least one mutation
        if num_mutations == 0:

            forced_idx = self.rng.integers(self.genome_size)

            mutation_mask[forced_idx] = True

            num_mutations = 1

        # ------------------------------------------------------
        # Apply Gaussian mutations
        # ------------------------------------------------------
        child[mutation_mask] += self.rng.normal(
            loc=0.0,
            scale=mutation_strength,
            size=num_mutations
        )

        # ------------------------------------------------------
        # Clip to genome bounds
        # ------------------------------------------------------
        child = np.clip(
            child,
            self.genome_bounds[0],
            self.genome_bounds[1]
        )

        return child.astype(np.float32)

    # ==========================================================
    # NOISY EVALUATION SUPPORT
    # ==========================================================

    def evaluate_multiple(
        self,
        genome: np.ndarray,
        evaluation_function,
        num_evaluations: int = 5
    ) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate a genome multiple times to reduce noise.

        evaluation_function must return:
            fitness, behavior_descriptors
        """

        fitnesses = []
        bds = []

        for _ in range(num_evaluations):

            fitness, bd = evaluation_function(genome)

            fitnesses.append(fitness)
            bds.append(bd)

            self.num_evaluations += 1

        fitnesses = np.array(fitnesses)

        # Mean BD across evaluations
        mean_bd = np.mean(np.array(bds), axis=0)

        return (
            float(np.mean(fitnesses)),
            float(np.std(fitnesses)),
            mean_bd
        )

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def get_total_cells(self) -> int:

        return int(np.prod(self.grid_shape))

    def get_statistics(self) -> Dict:

        if not self.grid:

            return {
                'coverage': 0.0,
                'num_filled': 0,
                'max_fitness': 0.0,
                'mean_fitness': 0.0,
                'qd_score': 0.0
            }

        fitnesses = np.array([
            cell['fitness']
            for cell in self.grid.values()
        ])

        return {

            'coverage':
                len(self.grid) / self.get_total_cells(),

            'num_filled':
                len(self.grid),

            'max_fitness':
                float(np.max(fitnesses)),

            'mean_fitness':
                float(np.mean(fitnesses)),

            'qd_score':
                float(np.sum(fitnesses))
        }

    def update_history(self):

        stats = self.get_statistics()

        self.history['coverage'].append(stats['coverage'])
        self.history['max_fitness'].append(stats['max_fitness'])
        self.history['mean_fitness'].append(stats['mean_fitness'])
        self.history['qd_score'].append(stats['qd_score'])
        self.history['evaluations'].append(self.num_evaluations)

    # ==========================================================
    # IO
    # ==========================================================

    def save(self, filepath: str):

        with open(filepath, 'wb') as f:

            pickle.dump({

                'grid': self.grid,

                'bd_dims': self.bd_dims,

                'genome_size': self.genome_size,

                'genome_bounds': self.genome_bounds,

                'bd_handling': self.bd_handling,

                'num_evaluations': self.num_evaluations,

                'invalid_bd_count': self.invalid_bd_count,

                'history': self.history,

                'grid_shape': self.grid_shape,

                'seed': self.seed,

                'bd_mode': self.bd_mode,

                'fitness_mode': self.fitness_mode,

                'bd_labels': self.bd_labels,

                'fitness_label': self.fitness_label

            }, f)

    def load(self, filepath: str):

        with open(filepath, 'rb') as f:

            data = pickle.load(f)

        self.grid = data['grid']

        self.bd_dims = data['bd_dims']

        self.genome_size = data['genome_size']

        self.genome_bounds = data['genome_bounds']

        self.bd_handling = data.get('bd_handling', 'reject')

        self.num_evaluations = data['num_evaluations']

        self.invalid_bd_count = data.get(
            'invalid_bd_count',
            0
        )

        self.history = data['history']

        self.grid_shape = data['grid_shape']

        self.seed = data.get('seed', None)

        # Restore RNG
        self.rng = np.random.default_rng(self.seed)

        self.bd_mode = data.get('bd_mode', None)

        self.fitness_mode = data.get('fitness_mode', None)

        self.bd_labels = data.get('bd_labels', None)

        self.fitness_label = data.get('fitness_label', None)

    # ==========================================================
    # ACCESS
    # ==========================================================

    def get_best_per_bin(self) -> Dict:

        return self.grid.copy()