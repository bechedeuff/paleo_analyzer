from typing import List, Dict, Any
import os
import json
from datetime import datetime

# Import configurations
from . import configurations as config


def validate_configurations() -> List[str]:
    """
    Validate configuration parameters and show warnings if needed.
    
    Returns:
    --------
    List[str]
        List of warning messages for invalid configurations
    """
    warnings: List[str] = []
    
    # Check if data files exist
    proxy1_path: str = f'data/{config.PROXY1_FILE}'
    proxy2_path: str = f'data/{config.PROXY2_FILE}'
    
    if not os.path.exists(proxy1_path):
        warnings.append(f"⚠️  Proxy 1 file not found: {proxy1_path}")
    
    if not os.path.exists(proxy2_path):
        warnings.append(f"⚠️  Proxy 2 file not found: {proxy2_path}")
    
    # Check window size
    if config.WINDOW_SIZE < 5:
        warnings.append(f"⚠️  Small window size ({config.WINDOW_SIZE} kyr) may generate unstable results")
    elif config.WINDOW_SIZE > 500:
        warnings.append(f"⚠️  Large window size ({config.WINDOW_SIZE} kyr) may lose temporal details")
    
    # Check thresholds
    if not (0 < config.THRESHOLD_HIGH <= 1):
        warnings.append(f"⚠️  THRESHOLD_HIGH should be between 0 and 1 (current: {config.THRESHOLD_HIGH})")
    
    if not (0 < config.THRESHOLD_LOW < config.THRESHOLD_HIGH):
        warnings.append(f"⚠️  THRESHOLD_LOW should be between 0 and THRESHOLD_HIGH (current: {config.THRESHOLD_LOW})")
    
    # Check window comparison sizes
    if len(config.WINDOW_COMPARISON['window_sizes']) != len(config.WINDOW_COMPARISON['cycle_names']):
        warnings.append(f"⚠️  Number of window_sizes and cycle_names should match")
    
    return warnings


def print_configuration_summary() -> None:
    """
    Print a summary of current configurations.
    """
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Proxy 1 file: {config.PROXY1_FILE}")
    print(f"Proxy 2 file: {config.PROXY2_FILE}")
    print(f"Window size: {config.WINDOW_SIZE} kyr")
    print(f"Interpolation resolution: {config.INTERPOLATION_RESOLUTION} kyr")
    print(f"Correlation thresholds: High={config.THRESHOLD_HIGH}, Low={config.THRESHOLD_LOW}")
    print(f"X-axis tick interval: {config.TEMPORAL_EVOLUTION['x_tick_interval']} kyr")
    print(f"Window comparison sizes: {config.WINDOW_COMPARISON['window_sizes']} kyr")
    print()


def create_results_directory() -> str:
    """
    Create a new experiment directory with sequential numbering.
    
    Returns:
    --------
    str
        Path to the created experiment directory
    """
    # Ensure main results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Find the next experiment number
    experiment_num: int = 1
    while True:
        experiment_dir: str = f'results/experiment_{experiment_num}'
        if not os.path.exists(experiment_dir):
            break
        experiment_num += 1
    
    # Create the experiment directory and subdirectories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f'{experiment_dir}/figures', exist_ok=True)
    
    print(f"📁 Created experiment directory: {experiment_dir}")
    return experiment_dir


def save_experiment_config(experiment_dir: str) -> str:
    """
    Save the current configuration to the experiment directory for reproducibility.
    
    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    config_data: Dict[str, Any] = {
        "experiment_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "configuration_used": "src/configurations.py"
        },
        "data_files": {
            "proxy1_file": config.PROXY1_FILE,
            "proxy2_file": config.PROXY2_FILE
        },
        "analysis_parameters": {
            "window_size": config.WINDOW_SIZE,
            "interpolation_resolution": config.INTERPOLATION_RESOLUTION,
            "min_periods": config.MIN_PERIODS,
            "threshold_high": config.THRESHOLD_HIGH,
            "threshold_low": config.THRESHOLD_LOW
        },
        "visualization_settings": {
            "matplotlib_style": config.MATPLOTLIB_STYLE,
            "seaborn_palette": config.SEABORN_PALETTE,
            "default_figure_size": config.DEFAULT_FIGURE_SIZE,
            "default_font_size": config.DEFAULT_FONT_SIZE
        },
        "plot_configurations": {
            "comprehensive_analysis": config.COMPREHENSIVE_ANALYSIS,
            "temporal_evolution": config.TEMPORAL_EVOLUTION,
            "window_comparison": config.WINDOW_COMPARISON
        }
    }
    
    config_file: str = f'{experiment_dir}/experiment_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Experiment configuration saved: {config_file}")
    return config_file