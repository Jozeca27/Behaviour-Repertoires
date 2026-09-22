import numpy as np
from typing import List, Tuple
import pickle
from cmaes import CMA


class CMAEmitter:
    def __init__(self, mean, sigma, bounds, population_size):
        self.dim = len(mean)
        self.bounds = bounds
        self.initial_sigma = sigma
        self.population_size = population_size

        bounds_array = np.array([[bounds[0], bounds[1]]] * self.dim)

        self.optimizer = CMA(
            mean=mean,
            sigma=sigma,
            bounds=bounds_array,
            population_size=population_size
        )

        self.no_improvement_steps = 0

    # ==============================
    # ASK
    # ==============================
    def ask(self):
        """
        Sample a batch of solutions from CMA-ES
        """
        solutions = np.array([self.optimizer.ask() for _ in range(self.population_size)])

        # Clip to bounds (safety)
        solutions = np.clip(solutions, self.bounds[0], self.bounds[1])

        return solutions

    # ==============================
    # TELL
    # ==============================
    def tell(self, solutions, fitnesses):
        """
        fitnesses = improvement signal (NOT raw fitness)
        CMA minimizes → we negate improvements
        """

        # Convert to CMA format (NO tolist!)
        data = [(s, -f) for s, f in zip(solutions, fitnesses)]
        self.optimizer.tell(data)

        # Restart logic based on actual improvement
        EPS = 1e-12
        if np.any(fitnesses > EPS):
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1

    # ==============================
    # RESTART LOGIC
    # ==============================
    def should_restart(self, patience):
        return self.no_improvement_steps > patience

    def restart(self, new_mean):
        """
        Full CMA reset (important for correctness)
        """
        bounds_array = np.array([[self.bounds[0], self.bounds[1]]] * self.dim)

        self.optimizer = CMA(
            mean=new_mean,
            sigma=self.initial_sigma,
            bounds=bounds_array,
            population_size=self.population_size
        )

        self.no_improvement_steps = 0

class CMAMAE:
    def __init__(
        self,
        behavior_descriptors_dims: List[Tuple[float, float, int]],
        genome_size: int,
        genome_bounds=(-1, 1),
        num_emitters=10,
        emitter_batch_size=64,
        initial_sigma=0.3,
        emitter_restart_patience=200,
        annealing_schedule=None
    ):
        self.bd_dims = behavior_descriptors_dims
        self.genome_size = genome_size
        self.genome_bounds = genome_bounds

        self.num_emitters = num_emitters
        self.batch_size = emitter_batch_size
        self.initial_sigma = initial_sigma
        self.restart_patience = emitter_restart_patience

        self.bd_mode = None
        self.fitness_mode = None
        self.bd_labels = None
        self.fitness_label = None

        self.grid = {}
        self.grid_shape = tuple([d[2] for d in self.bd_dims])

        self.emitters = []
        self.num_evaluations = 0

        # Annealing schedule
        if annealing_schedule is None:
            # Default annealing schedule (assumes 2D BD)
            self.annealing_stages = [
                (0, [5, 5]),
                (2000, [10, 10]),
                (5000, [20, 20]),
            ]
        else:
            self.annealing_stages = annealing_schedule
        self.current_stage = 0

    # ---------- GRID ----------
    def get_grid_index(self, bd):
        idx = []
        for val, (low, high, bins) in zip(bd, self.bd_dims):
            if val < low or val > high:
                return None
            i = int((val - low) / (high - low) * bins)
            i = min(i, bins - 1)
            idx.append(i)
        return tuple(idx)

    def add_to_grid(self, genome, fitness, raw_bd, norm_bd):
        idx = self.get_grid_index(raw_bd)
        if idx is None:
            return False

        if idx not in self.grid or self.grid[idx]['fitness'] < fitness:
            self.grid[idx] = {
                'genome': genome.copy(),
                'fitness': fitness,
                'bd': norm_bd.copy(),
                'raw_bd': raw_bd.copy()
            }
            return True
        return False

    def random_genome(self):
        return np.random.uniform(
            self.genome_bounds[0],
            self.genome_bounds[1],
            self.genome_size
        )

    def get_total_cells(self):
        if hasattr(self, 'bd_dims') and self.bd_dims:
            try:
                return int(np.prod(tuple(int(dim[2]) for dim in self.bd_dims)))
            except Exception:
                pass
        return int(np.prod(self.grid_shape))

    def sample_elite_weighted(self):
        if not self.grid:
            return self.random_genome()

        elites = list(self.grid.values())
        fitnesses = np.array([e['fitness'] for e in elites])

        probs = fitnesses - fitnesses.min() + 1e-6
        probs /= probs.sum()

        idx = np.random.choice(len(elites), p=probs)
        return elites[idx]['genome']

    # ---------- EMITTERS ----------
    def create_emitters(self):
        self.emitters = []
        for _ in range(self.num_emitters):
            mean = self.sample_elite_weighted()
            emitter = CMAEmitter(
                mean,
                self.initial_sigma,
                self.genome_bounds,
                self.batch_size
            )
            self.emitters.append(emitter)

    def ask(self):
        return np.vstack([e.ask() for e in self.emitters])

    def tell(self, solutions, fitnesses, raw_bds, norm_bds):
        ptr = 0

        for emitter in self.emitters:
            sols = solutions[ptr:ptr+self.batch_size]
            fits = fitnesses[ptr:ptr+self.batch_size]
            raw_desc = raw_bds[ptr:ptr+self.batch_size]
            norm_desc = norm_bds[ptr:ptr+self.batch_size]

            # Compute novelty bonus to encourage exploration
            novelty_fits = []
            for s, f, raw_bd, norm_bd in zip(sols, fits, raw_desc, norm_desc):
                novelty_bonus = self._compute_novelty_bonus(norm_bd)
                combined_fitness = f + novelty_bonus
                novelty_fits.append(combined_fitness)
                
                self.add_to_grid(s, f, raw_bd, norm_bd)

            # Tell emitter about combined fitness (original + novelty bonus)
            emitter.tell(sols, np.array(novelty_fits))

            if emitter.should_restart(self.restart_patience):
                emitter.restart(self.sample_elite_weighted())

            ptr += self.batch_size

        self.num_evaluations += len(solutions)
        self.check_annealing()

    def _compute_novelty_bonus(self, bd, k_neighbors=15):
        """
        Compute novelty bonus based on distance to nearest neighbors in behavior space
        Rewards BDs that reach underexplored regions
        
        Args:
            bd: Behavior descriptor (normalized to [0,1])
            k_neighbors: Number of neighbors to consider
            
        Returns:
            Novelty bonus (higher = more novel)
        """
        if len(self.grid) == 0:
            return 1.0  # Early solutions get max novelty bonus
        
        # Compute distances to all existing BDs in grid
        grid_bds = np.array([v['bd'] for v in self.grid.values()])
        
        # Check for dimension mismatch
        expected_dim = len(self.bd_dims)
        actual_dim = bd.shape[0] if len(bd.shape) > 0 else 1
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"Behavior descriptor shape mismatch: expected {expected_dim}D (from {len(self.grid)} grid entries), "
                f"got {actual_dim}D. This typically means: (1) BD_MODE in config.py changed between runs, "
                f"(2) old results in 'results/' folder from a run with different settings, or "
                f"(3) behavior descriptor computation is returning wrong shape. "
                f"SOLUTION: Delete contents of 'results/' folder and restart."
            )
        
        # Euclidean distance in behavior space
        distances = np.linalg.norm(grid_bds - bd, axis=1)
        
        # Get k-nearest neighbor distance (novelty metric)
        k = min(k_neighbors, len(distances))
        knn_distance = np.mean(np.sort(distances)[:k])
        
        # Normalize novelty bonus to [0, 1]
        # Grid is normalized to [0,1], so max distance is sqrt(d)
        max_distance = np.sqrt(len(self.bd_dims))
        novelty_bonus = (knn_distance / max_distance) * 0.5  # Scale to avoid dominating fitness
        
        return novelty_bonus

    # ---------- ANNEALING ----------
    def check_annealing(self):
        if self.current_stage < len(self.annealing_stages) - 1:
            next_eval, new_bins = self.annealing_stages[self.current_stage + 1]
            if self.num_evaluations >= next_eval:
                self.current_stage += 1
                self.update_resolution(new_bins)

    def update_resolution(self, bins_per_dim):
        if len(bins_per_dim) != len(self.bd_dims):
            if len(bins_per_dim) > len(self.bd_dims):
                print(
                    f"Warning: annealing schedule provides {len(bins_per_dim)} bins, "
                    f"but bd_dims has {len(self.bd_dims)} dimensions. "
                    f"Using the first {len(self.bd_dims)} entries."
                )
                bins_per_dim = bins_per_dim[:len(self.bd_dims)]
            else:
                raise ValueError(
                    f"Annealing schedule length {len(bins_per_dim)} does not match "
                    f"behavior descriptor dimensionality {len(self.bd_dims)}."
                )

        new_grid = {}

        self.bd_dims = [
            (low, high, bins)
            for (low, high, _), bins in zip(self.bd_dims, bins_per_dim)
        ]

        for cell in self.grid.values():
            idx = self.get_grid_index(cell['raw_bd'])
            if idx is not None:
                if idx not in new_grid or new_grid[idx]['fitness'] < cell['fitness']:
                    new_grid[idx] = cell

        self.grid = new_grid
        self.grid_shape = tuple(bins_per_dim)

    # ---------- STATS ----------
    def get_statistics(self):
        if not self.grid:
            return {
                "coverage": 0,
                "num_filled": 0,
                "qd_score": 0,
                "max_fitness": 0,
                "mean_fitness": 0
            }

        fitnesses = [v['fitness'] for v in self.grid.values()]
        total = self.get_total_cells()
        num_filled = len(self.grid)

        return {
            "coverage": num_filled / total,
            "num_filled": num_filled,
            "qd_score": np.sum(fitnesses),
            "max_fitness": np.max(fitnesses),
            "mean_fitness": np.mean(fitnesses)
        }