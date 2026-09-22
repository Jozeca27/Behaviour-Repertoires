"""
Configuration file for MAP-Elites Hexapod experiment
"""
import os
import numpy as np
from controllers.hexapod_goal_controller import GoalDirectedRecurrentTorqueHexapodController
from controllers.hexapod_controller_final import RecurrentTorqueHexapodController

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(PROJECT_ROOT, "robot_models", "hexapod.urdf")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
#CONTROLLER_GENOME_SIZE = GoalDirectedRecurrentTorqueHexapodController.get_genome_size()
CONTROLLER_GENOME_SIZE = RecurrentTorqueHexapodController.get_genome_size()

# Shared genome bounds for MAP-Elites runs
GENOME_BOUND_LOW = -0.75
GENOME_BOUND_HIGH = 0.75
GENOME_BOUNDS = (GENOME_BOUND_LOW, GENOME_BOUND_HIGH)

# Fitness weights for combined fitness mode: fitness = (DISTANCE_WEIGHT * distance + AVG_SPEED_WEIGHT * avg_speed - STABILITY_PENALTY_WEIGHT * stability_penalty - ENERGY_WEIGHT * energy)
DIRECTION_WEIGHT = 3.0
DISTANCE_WEIGHT = 5.0 #3.0
AVG_SPEED_WEIGHT = 0.8 #0.3
STABILITY_PENALTY_WEIGHT = 0.05 #0.2
ENERGY_WEIGHT = 2e-8 #2e-6

# High-level behaviour descriptor and fitness selection
# BD_MODE options: 'xy_position', 'duty_factor', 'transformer_latent', 'transformer_8d'
BD_MODE = 'xy_position'
# FITNESS_MODE options: 'combined' (default), 'distance', 'forward_velocity', 'avg_speed', 'stability'
FITNESS_MODE = 'distance'

# Transformer-based BD configurations
_TRANSFORMER_COMMON_CONFIG = {
    # Model architecture used during training; must match checkpoint
    'state_dim': 73,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'max_seq_len': 2000,
    # sequence length
    'sequence_length': 256,
    # Optional device override: 'auto', 'cpu', or 'cuda'
    'device': 'auto',
}

TRANSFORMER_BD_CONFIGS = {
    'transformer_latent': {
        **_TRANSFORMER_COMMON_CONFIG,
        # Relative to PROJECT_ROOT unless absolute path is provided
        'model_path': os.path.join('results', 'transformer', 'hexapod_transformer_fitness.pt'),
        # Which latent coordinates become the BD dimensions
        'latent_indices': [0, 1],
        # (min, max, bins) per selected latent coordinate
        'bd_ranges': [(-5.0, 5.0, 50), (-5.0, 5.0, 50)],
        'bd_dim': 2,
    },
    'transformer_8d': {
        **_TRANSFORMER_COMMON_CONFIG,
        # Relative to PROJECT_ROOT unless absolute path is provided
        'model_path': os.path.join('results', 'transformer', 'hexapod_transformer_fitness_8D.pt'),
        # Number of learned descriptor outputs
        'bd_dim': 8,
        # Explicit ranges for each learned descriptor dimension
        'bd_ranges': [(-10.0, 10.0, 6) for _ in range(8)],
    },
}


def get_transformer_bd_config(mode=None):
    """Return the transformer BD config for a specific BD mode."""
    if mode is None:
        mode = BD_MODE

    if mode not in TRANSFORMER_BD_CONFIGS:
        raise ValueError(f"Unknown transformer BD mode '{mode}'")

    return TRANSFORMER_BD_CONFIGS[mode]


TRANSFORMER_BD_CONFIG = (
    get_transformer_bd_config()
    if BD_MODE in TRANSFORMER_BD_CONFIGS
    else None
)

# Backwards-compatibility aliases for callers that want an explicit name.
TRANSFORMER_LATENT_BD_CONFIG = TRANSFORMER_BD_CONFIGS['transformer_latent']
TRANSFORMER_8D_BD_CONFIG = TRANSFORMER_BD_CONFIGS['transformer_8d']

def get_behavior_descriptor_dims(mode=None):
    """Get behavior descriptor ranges and bins for a BD_MODE"""
    if mode is None:
        mode = BD_MODE

    if mode == 'xy_position':
        return [
            #(-3.0, 3.0, 8), (-3.0, 3.0, 8),  # forward
            #(-3.0, 3.0, 8), (-3.0, 3.0, 8),  # left
            #(-3.0, 3.0, 8), (-3.0, 3.0, 8),  # right
            (-5, 5, 50), (-5, 5, 50)
        ]
    elif mode == 'duty_factor':
        # One duty factor per leg, normalized in [0, 1]
        return [(0.0, 1.0, 8) for _ in range(6)]
    elif mode == 'transformer_latent':
        return get_transformer_bd_config('transformer_latent')['bd_ranges']
    elif mode == 'transformer_8d':
        return get_transformer_bd_config('transformer_8d')['bd_ranges']
    else:
        raise ValueError(f"Unknown BD_MODE '{mode}'")


def get_behavior_descriptor_names(mode=None):
    """Return readable behavior descriptor labels for a BD_MODE."""
    if mode is None:
        mode = BD_MODE

    if mode == 'xy_position':
        return [
            'Forward X',
            'Forward Y',
            'Left X',
            'Left Y',
            'Right X',
            'Right Y',
        ]
    elif mode == 'duty_factor':
        return [f"Duty Factor L{i}" for i in range(1, 7)]
    elif mode == 'transformer_latent':
        latent_indices = get_transformer_bd_config('transformer_latent').get('latent_indices', [])
        return [f"Latent {idx}" for idx in latent_indices]
    elif mode == 'transformer_8d':
        bd_dim = get_transformer_bd_config('transformer_8d').get('bd_dim', 8)
        return [f"BD {i}" for i in range(bd_dim)]
    else:
        raise ValueError(f"Unknown BD_MODE '{mode}'")


def get_fitness_name(mode=None):
    """Return a readable fitness label for FITNESS_MODE."""
    if mode is None:
        mode = FITNESS_MODE

    if mode == 'combined':
        return 'Locomotion Fitness'
    elif mode == 'distance':
        return 'Distance traveled'
    elif mode == 'forward_velocity':
        return 'Forward velocity'
    elif mode == 'avg_speed':
        return 'Average speed'
    elif mode == 'stability':
        return 'Stability score'
    else:
        raise ValueError(f"Unknown FITNESS_MODE '{mode}'")


def get_default_annealing_schedule(bd_dims=None):
    """Return a default annealing schedule based on BD dimensionality."""
    if bd_dims is None:
        bd_dims = get_behavior_descriptor_dims()

    dim = len(bd_dims)
    if dim == 2:
        return [
            (0, [5, 5]),
            (500, [10, 10]),
            (1000, [15, 15]),
            (1500, [20, 20]),
        ]
    elif dim == 4:
        return [
            (0, [5, 5, 4, 4]),
            (500, [10, 10, 8, 8]),
            (1000, [15, 15, 12, 12]),
            (1500, [20, 20, 15, 15]),
        ]
    elif dim == 6:
        return [
            (0,    [4, 4, 4, 4, 4, 4]),
            (500,  [6, 6, 6, 6, 6, 6]),
            (1000, [8, 8, 8, 8, 8, 8]),
            (1500, [10,10,10,10,10,10]),
        ]
    elif dim == 8:
        return [
            (0, [5, 5, 4, 4, 4, 4, 4, 4]),
            (500, [8, 8, 6, 6, 6, 6, 6, 6]),
            (1000, [10, 10, 8, 8, 8, 8, 8, 8]),
            (1500, [12, 12, 10, 10, 10, 10, 10, 10]),
        ]
    else:
        raise ValueError(f"Unsupported BD dimensionality {dim} for annealing schedule")

# MAP-Elites Configuration
MAP_ELITES_CONFIG = {
    # Behavior descriptor dimensions used for MAP-Elites
    'behavior_descriptors': get_behavior_descriptor_dims(),
    
    # Genome configuration
    'genome_size': CONTROLLER_GENOME_SIZE,
    'genome_bounds': GENOME_BOUNDS,
    
    # Evolution parameters
    'num_initial_random': 10000,
    'num_iterations': 5000,
    'batch_size': 64,  # Evaluate this many in parallel
    
    # Mutation parameters
    'mutation_rate': 0.002,
    'mutation_strength': 0.05,

    # BD handling
    # "reject" for geometric descriptors
    # "clip" for latent descriptors
    'bd_handling': 'reject',

    # Repeated evaluations
    'num_repeated_evaluations': 3,

    # Reproducibility
    'seed': 42,
}

# CMA-MAE Configuration
CMA_MAE_CONFIG = {
    # Behavior descriptor dimensions used for CMA-MAE
    'behavior_descriptors': get_behavior_descriptor_dims(),
    
    # Genome configuration
    'genome_size': CONTROLLER_GENOME_SIZE,
    'genome_bounds': (-2.0, 2.0),
    
    # CMA-ES Emitter parameters
    'num_emitters': 10,              # Number of CMA-ES emitters
    'emitter_batch_size': 64,       # Solutions per emitter per iteration
    'initial_sigma': 0.3,           # Initial step-size for CMA-ES
    'emitter_restart_patience': 200, # Generations without improvement before restart
    
    # Evolution parameters
    'num_initial_random': 100,      # Initial random solutions to seed archive
    'num_iterations': 100,        # Total iterations (each generates num_emitters * batch_size solutions)
    
    # MAP-Annealing schedule: (iteration, bins_per_dimension)
    # Gradually refines grid resolution over time
    # This is overridden below to match behavior descriptor dimensionality.
    'annealing_schedule': None,
}

# Set the default CMA-MAE annealing schedule for the configured BD dimensionality
CMA_MAE_CONFIG['annealing_schedule'] = get_default_annealing_schedule(CMA_MAE_CONFIG['behavior_descriptors'])

# Simulation Configuration
SIMULATION_CONFIG = {
    'duration': 6.0,  # Simulation time in seconds
    'time_step': 1./240,
    'gui': False,  # Set to True for visualization
    'max_force': 5.0,
    'sample_interval': 1.0,  # Sample leg positions every N seconds for behavior descriptors
    'acceleration_threshold': 20.0,  # Max acceleration (m/s²) before marking as invalid; prevents explosion-prone gaits
    'warmup_time': 1.0,
    'controller_interval': 4,
}

# Logging Configuration
LOGGING_CONFIG = {
    'save_interval': 500,  # Save grid every N iterations
    'trajectory_save_interval': 500,  # Save trajectory logs every N iterations
    'save_random_trajectories': False,  # Enable/disable random trajectory capture
    'trajectory_save_probability': 0.1,  # Probability of saving a random trajectory
    'print_interval': 1000,  # Print stats every N iterations
    'save_grid': True,
    'save_plots': True,
    'save_detailed_logs': False,  # Save detailed timestep logs for best controllers
    'num_best_to_log': 20,  # Number of best controllers to save detailed logs for
    'num_random_to_log': 20,  # Number of random controllers to save detailed logs for
}

GOAL_VECTORS = [
    np.array([1.0, 0.0, 0.0], dtype=np.float32),   # forward
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # left
    np.array([0.0, -1.0, 0.0], dtype=np.float32),  # right
]
