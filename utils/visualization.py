"""
Visualization utilities for MAP-Elites results
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import os


def infer_grid_shape(map_elites, warn=True):
    """Infer a 2D grid shape from the MAP-Elites object."""
    if hasattr(map_elites, 'bd_dims') and len(map_elites.bd_dims) == 2:
        try:
            derived_shape = tuple(int(dim[2]) for dim in map_elites.bd_dims)
            if warn and hasattr(map_elites, 'grid_shape') and tuple(map_elites.grid_shape) != derived_shape:
                print(
                    f"Warning: derived 2D grid_shape {derived_shape} from bd_dims "
                    f"differs from stored grid_shape {map_elites.grid_shape}. Using derived shape for plotting."
                )
            return derived_shape
        except Exception:
            pass

    if hasattr(map_elites, 'grid_shape'):
        try:
            return tuple(map_elites.grid_shape)
        except Exception:
            pass

    raise ValueError('Unable to infer grid_shape for plotting from the provided object.')


def infer_total_cells(map_elites):
    """Infer the total number of grid cells from the MAP-Elites object."""
    grid_shape = infer_grid_shape(map_elites)
    return int(np.prod(grid_shape))


def plot_map_elites_grid(map_elites, save_path=None, show=True):
    """
    Visualize the MAP-Elites grid as a heatmap
    
    Args:
        map_elites: MAPElites instance
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if len(map_elites.bd_dims) != 2:
        print("Visualization only supports 2D behavior descriptors")
        return
    
    # Create grid for visualization
    grid_shape = infer_grid_shape(map_elites)
    if len(grid_shape) != 2:
        print("Visualization only supports 2D behavior descriptors")
        return

    fitness_grid = np.full(grid_shape, np.nan, dtype=float)
    
    # Fill in fitness values
    for cell_idx, cell_data in map_elites.grid.items():
        fitness_grid[cell_idx] = cell_data['fitness']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get behavior descriptor ranges
    bd1_min, bd1_max, bd1_bins = map_elites.bd_dims[0]
    bd2_min, bd2_max, bd2_bins = map_elites.bd_dims[1]
    
    # Plot heatmap
    im = ax.imshow(
        fitness_grid.T,
        origin='lower',
        aspect='auto',
        extent=[bd1_min, bd1_max, bd2_min, bd2_max],
        cmap='viridis',
        interpolation='nearest'
    )
    
    # Colorbar label
    fitness_label = getattr(map_elites, 'fitness_label', None) or getattr(map_elites, 'fitness_mode', None) or 'Fitness'
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(fitness_label, rotation=270, labelpad=20)
    
    # Axis labels
    if hasattr(map_elites, 'bd_labels') and isinstance(map_elites.bd_labels, (list, tuple)):
        x_label = map_elites.bd_labels[0]
        y_label = map_elites.bd_labels[1] if len(map_elites.bd_labels) > 1 else 'Behavior dimension 2'
    else:
        x_label = 'Behavior dimension 1'
        y_label = 'Behavior dimension 2'

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    actual_coverage = len(map_elites.grid) / infer_total_cells(map_elites)
    title_mode = getattr(map_elites, 'bd_mode', None)
    title_suffix = f" ({title_mode})" if title_mode else ''
    ax.set_title(f'MAP-Elites Grid - Coverage: {actual_coverage*100:.1f}%{title_suffix}')
    
    # Grid
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_evolution_history(map_elites, save_path=None, show=True):
    """
    Plot evolution metrics over time
    
    Args:
        map_elites: MAPElites instance
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if not map_elites.history['evaluations']:
        print("No history data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    evals = map_elites.history['evaluations']
    
    # Coverage
    axes[0, 0].plot(evals, np.array(map_elites.history['coverage']) * 100)
    axes[0, 0].set_xlabel('Evaluations')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].set_title('Grid Coverage Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Max Fitness
    axes[0, 1].plot(evals, map_elites.history['max_fitness'], color='green')
    axes[0, 1].set_xlabel('Evaluations')
    axes[0, 1].set_ylabel('Max Fitness')
    axes[0, 1].set_title('Maximum Fitness Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean Fitness
    axes[1, 0].plot(evals, map_elites.history['mean_fitness'], color='orange')
    axes[1, 0].set_xlabel('Evaluations')
    axes[1, 0].set_ylabel('Mean Fitness')
    axes[1, 0].set_title('Mean Fitness Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # QD Score
    total_cells = np.prod(infer_grid_shape(map_elites))
    qd_scores = [
        np.array(map_elites.history['mean_fitness'][i]) *
        np.array(map_elites.history['coverage'][i]) *
        total_cells
        for i in range(len(evals))
    ]
    axes[1, 1].plot(evals, qd_scores, color='red')
    axes[1, 1].set_xlabel('Evaluations')
    axes[1, 1].set_ylabel('QD-Score')
    axes[1, 1].set_title('Quality-Diversity Score Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_best_solutions(map_elites, simulation, num_solutions=5):
    """
    Run and visualize the best N solutions
    
    Args:
        map_elites: MAPElites instance
        simulation: HexapodSimulation instance with GUI enabled
        num_solutions: Number of solutions to visualize
    """
    # Get cells sorted by fitness
    cells = [(idx, data) for idx, data in map_elites.grid.items()]
    cells.sort(key=lambda x: x[1]['fitness'], reverse=True)
    
    print(f"\nVisualizing top {num_solutions} solutions:")
    
    for i, (cell_idx, cell_data) in enumerate(cells[:num_solutions]):
        print(f"\n{i+1}. Cell {cell_idx}")
        print(f"   Fitness: {cell_data['fitness']:.4f}")
        print(f"   Behavior: {cell_data['bd']}")
        
        # Run simulation with visualization
        result = simulation.evaluate_controller(
            cell_data['genome'],
            duration=10.0,
            render=True
        )
        
        input("Press Enter to continue to next solution...")


def export_grid_to_csv(map_elites, filepath):
    """
    Export grid data to CSV for external analysis
    
    Args:
        map_elites: MAPElites instance
        filepath: Path to save CSV
    """
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['cell_indices', 'fitness'] + \
                 [f'bd_{i}' for i in range(len(map_elites.bd_dims))] + \
                 [f'gene_{i}' for i in range(map_elites.genome_size)]
        writer.writerow(header)
        
        # Data
        for cell_idx, cell_data in map_elites.grid.items():
            row = [str(cell_idx), cell_data['fitness']] + \
                  list(cell_data['bd']) + \
                  list(cell_data['genome'])
            writer.writerow(row)
    
    print(f"Exported grid to: {filepath}")
