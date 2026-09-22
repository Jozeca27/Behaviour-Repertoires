# MAP-Elites for Hexapod Locomotion

Quality Diversity framework for evolving diverse hexapod gaits using PyBullet physics simulation.

## Overview

This project implements MAP-Elites algorithm to discover a diverse repertoire of locomotion behaviors for a 6-legged robot. The algorithm maintains a grid where each cell contains the best controller found for a specific behavior characteristic (e.g., final XY position).

## Project Structure

```
.
├── robot_models/
│   └── hexapod.urdf          # Hexapod robot description
├── controllers/
│   └── hexapod_controller.py # Neural-network policy controller
├── map_elites/
│   └── map_elites.py         # MAP-Elites algorithm
├── utils/                     # Utility functions (visualization, analysis)
├── results/                   # Output files (grids, plots)
├── simulation.py             # PyBullet simulation environment
├── config.py                 # Configuration parameters
├── main.py                   # Main execution script
└── requirements.txt          # Python dependencies
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Run

```bash
python main.py
```

This will:
- Initialize the MAP-Elites grid
- Generate 100 random controllers
- Evolve for 10,000 iterations
- Save results to `results/` directory

### Configuration

Edit `config.py` to customize:

**Behavior descriptor and fitness modes:**
```python
BD_MODE = 'xy_position'        # options: 'leg_position', 'xy_position', 'xy_displacement'
FITNESS_MODE = 'combined'      # options: 'combined', 'distance', 'forward_velocity', 'avg_speed', 'stability'
```

The default `xy_position` mode uses final robot (x, y) as the BD, and `combined` uses weighted distance + velocity - stability.

**Behavior Descriptors:**
```python
'behavior_descriptors': [
    (-2.0, 2.0, 20),  # Final X position (min, max, bins)
    (-2.0, 2.0, 20),  # Final Y position
]
```

**Evolution Parameters:**
```python
'num_iterations': 10000,
'mutation_rate': 0.1,
'mutation_strength': 0.2,
```

**Visualization:**
```python
'gui': True,  # Show PyBullet GUI during evolution
```

### Visualize Results

After running, you can visualize the best solutions:

```python
from map_elites.map_elites import MAPElites
from simulation import HexapodSimulation
import config

# Load saved grid
map_elites = MAPElites(...)
map_elites.load('results/map_elites_final.pkl')

# Visualize a specific cell
sim = HexapodSimulation(config.URDF_PATH, gui=True)
cell = map_elites.grid[(10, 10)]  # Example cell
result = sim.evaluate_controller(cell['genome'], duration=10.0, render=True)
```

## Controller Encoding

The hexapod controller uses a feedforward neural network policy.

- Input: joint positions, joint velocities, base linear/angular velocity, base orientation (Euler), and periodic time features.
- Output: 18 target joint angles (6 legs x 3 joints), centered around a stable standing pose.
- Genome: flattened NN weights and biases.

The exact genome size is computed automatically from the architecture via HexapodController.get_genome_size().

## Behavior Descriptors

Current implementation uses:
- **Final XY position**: Where the robot ends up
- **Duty Factor**: How the legs behave during the run

## Fitness Function

Fitness = Euclidean distance traveled from start position

Penalized if robot falls over (base height < 5cm)

## Extending the Framework

### Custom Behavior Descriptors

Edit `simulation.py`:

```python
# In evaluate_controller()
bd_velocity = np.mean([np.linalg.norm(v) for v in velocities])
bd_turning = calculate_turning_metric(positions)
return {
    'fitness': distance,
    'behavior_descriptors': np.array([bd_velocity, bd_turning]),
    ...
}
```

### Different Controller

Implement in `controllers/`:

```python
class MyController:
    def __init__(self, genome):
        ...
    
    def get_joint_angles(self, t):
        # Your control logic
        return angles
```

## Results Interpretation

- **Coverage**: Percentage of grid cells filled with solutions
- **QD-Score**: Sum of all fitnesses in the grid (quality + diversity)
- **Max Fitness**: Best performing controller found
- **Mean Fitness**: Average performance across all cells

## Troubleshooting

**Robot falls immediately:**
- Adjust initial height in `simulation.py`
- Check URDF collision geometries
- Increase mutation strength to explore more

**Poor coverage:**
- Increase number of iterations
- Adjust behavior descriptor ranges
- Increase mutation rate

**Slow execution:**
- Set `gui=False` in config
- Reduce simulation duration
- Use fewer grid bins

## References

- Mouret & Clune (2015). "Illuminating the Space of Behavioral Features by Deconstructing Quality Diversity"
- Cully et al. (2015). "Robots that can adapt like animals"

## License

MIT License
