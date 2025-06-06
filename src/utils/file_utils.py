import os
import json
from typing import Dict, Any
from datetime import datetime

from src import config

def create_results_directory() -> str:
    """
    Create a new experiment directory with sequential numbering and subdirectories for the analyses.
    
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
    
    # Create the experiment directory and subdirectories for the analyses
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Rolling window analysis subdirectories
    os.makedirs(f'{experiment_dir}/rolling_window', exist_ok=True)
    os.makedirs(f'{experiment_dir}/rolling_window/figures', exist_ok=True)
    
    # Spectral analysis subdirectories
    os.makedirs(f'{experiment_dir}/spectral', exist_ok=True)
    os.makedirs(f'{experiment_dir}/spectral/figures', exist_ok=True)
    
    # Lead-lag analysis subdirectories
    os.makedirs(f'{experiment_dir}/lead_lag', exist_ok=True)
    os.makedirs(f'{experiment_dir}/lead_lag/figures', exist_ok=True)
    
    print(f"📁 Created experiment directory: {experiment_dir}")
    print(f"   ├── rolling_window/")
    print(f"   │   └── figures/")
    print(f"   ├── spectral/")
    print(f"   │   └── figures/")
    print(f"   ├── lead_lag/")
    print(f"   │   └── figures/")
    print(f"   └── experiment_config.json (will be created)")
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
        "rolling_window_analysis": config.ROLLING_WINDOW_ANALYSIS,
        "visualization_settings": {
            "matplotlib_style": config.MATPLOTLIB_STYLE,
            "seaborn_palette": config.SEABORN_PALETTE,
            "default_figure_size": config.DEFAULT_FIGURE_SIZE,
            "default_font_size": config.DEFAULT_FONT_SIZE
        },
        "rolling_window_plots": config.ROLLING_WINDOW_PLOTS,
        "spectral_analysis": config.SPECTRAL_ANALYSIS,
        "spectral_plots": config.SPECTRAL_PLOTS,
        "leadlag_analysis": config.LEADLAG_ANALYSIS,
        "leadlag_plots": config.LEADLAG_PLOTS
    }
    
    config_file: str = f'{experiment_dir}/experiment_config.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Experiment configuration saved: {config_file}")
    return config_file