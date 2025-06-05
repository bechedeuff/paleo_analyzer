from typing import List, Dict, Any, Tuple, Optional
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

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
    if config.ROLLING_WINDOW_ANALYSIS['window_size'] < 5:
        warnings.append(f"⚠️  Small window size ({config.ROLLING_WINDOW_ANALYSIS['window_size']} kyr) may generate unstable results")
    elif config.ROLLING_WINDOW_ANALYSIS['window_size'] > 500:
        warnings.append(f"⚠️  Large window size ({config.ROLLING_WINDOW_ANALYSIS['window_size']} kyr) may lose temporal details")
    
    # Check thresholds
    if not (0 < config.ROLLING_WINDOW_ANALYSIS['threshold_high'] <= 1):
        warnings.append(f"⚠️  THRESHOLD_HIGH should be between 0 and 1 (current: {config.ROLLING_WINDOW_ANALYSIS['threshold_high']})")
    
    if not (0 < config.ROLLING_WINDOW_ANALYSIS['threshold_low'] < config.ROLLING_WINDOW_ANALYSIS['threshold_high']):
        warnings.append(f"⚠️  THRESHOLD_LOW should be between 0 and THRESHOLD_HIGH (current: {config.ROLLING_WINDOW_ANALYSIS['threshold_low']})")
    
    # Check window comparison sizes
    if len(config.ROLLING_WINDOW_ANALYSIS['window_comparison_sizes']) != len(config.ROLLING_WINDOW_ANALYSIS['cycle_names']):
        warnings.append(f"⚠️  Number of window_sizes and cycle_names should match")
    
    # Check spectral analysis parameters
    if config.SPECTRAL_ANALYSIS['min_period'] <= 0:
        warnings.append(f"⚠️  MIN_PERIOD should be positive (current: {config.SPECTRAL_ANALYSIS['min_period']})")
    
    if config.SPECTRAL_ANALYSIS['max_period'] <= config.SPECTRAL_ANALYSIS['min_period']:
        warnings.append(f"⚠️  MAX_PERIOD should be greater than MIN_PERIOD")
    
    if not (0 < config.SPECTRAL_ANALYSIS['coherence_threshold'] <= 1):
        warnings.append(f"⚠️  COHERENCE_THRESHOLD should be between 0 and 1 (current: {config.SPECTRAL_ANALYSIS['coherence_threshold']})")
    
    if not (0 < config.SPECTRAL_ANALYSIS['confidence_level'] < 1):
        warnings.append(f"⚠️  CONFIDENCE_LEVEL should be between 0 and 1 (current: {config.SPECTRAL_ANALYSIS['confidence_level']})")
    
    if config.SPECTRAL_ANALYSIS['wavelet_param'] <= 0:
        warnings.append(f"⚠️  WAVELET_PARAM should be positive (current: {config.SPECTRAL_ANALYSIS['wavelet_param']})")
    
    return warnings

def print_configuration_summary() -> None:
    """
    Print a summary of current configurations.
    """
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 50)
    print("DATA FILES:")
    print(f"  Proxy 1 file: {config.PROXY1_FILE}")
    print(f"  Proxy 2 file: {config.PROXY2_FILE}")
    print(f"  Interpolation resolution: {config.INTERPOLATION_RESOLUTION} kyr")
    print()
    print("\nROLLING WINDOW ANALYSIS:")
    print(f"  Window size: {config.ROLLING_WINDOW_ANALYSIS['window_size']} kyr")
    print(f"  Correlation thresholds: High={config.ROLLING_WINDOW_ANALYSIS['threshold_high']}, Low={config.ROLLING_WINDOW_ANALYSIS['threshold_low']}")
    print(f"  Window comparison sizes: {config.ROLLING_WINDOW_ANALYSIS['window_comparison_sizes']} kyr")
    print("\nSPECTRAL ANALYSIS:")
    print(f"  Wavelet type: {config.SPECTRAL_ANALYSIS['wavelet_type']} (param={config.SPECTRAL_ANALYSIS['wavelet_param']})")
    print(f"  Period range: {config.SPECTRAL_ANALYSIS['min_period']} - {config.SPECTRAL_ANALYSIS['max_period']} kyr")
    print(f"  Coherence threshold: {config.SPECTRAL_ANALYSIS['coherence_threshold']}")
    print(f"  Confidence level: {config.SPECTRAL_ANALYSIS['confidence_level']}")
    print(f"  Milankovitch cycles: {list(config.SPECTRAL_ANALYSIS['milankovitch_cycles'].keys())}")
    print("\nLEAD-LAG ANALYSIS:")
    print(f"  Max lag: {config.LEADLAG_ANALYSIS['max_lag_kyr']} kyr")
    print(f"  Lag step: {config.LEADLAG_ANALYSIS['lag_step_kyr']} kyr")
    print(f"  Methods: {config.LEADLAG_ANALYSIS['methods']}")
    print(f"  Correlation types: {config.LEADLAG_ANALYSIS['correlation_types']}")
    print(f"  Bootstrap iterations: {config.LEADLAG_ANALYSIS['bootstrap_iterations']}")
    print()

def create_results_directory() -> str:
    """
    Create a new experiment directory with sequential numbering and subdirectories for both analyses.
    
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
    
    # Create the experiment directory and subdirectories for both analyses
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

def load_paleoclimate_data(proxy1_file: str, proxy2_file: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load and clean paleoclimate data from CSV files
    
    Parameters:
    -----------
    proxy1_file : str
        Path to CSV file of the first proxy
    proxy2_file : str
        Path to CSV file of the second proxy
        
    Expected format of CSVs:
    - First column: age in kyr
    - Second column: proxy values
        
    Returns:
    --------
    tuple : (success: bool, data: dict or None)
        - success: True if loading successful, False otherwise
        - data: Dictionary with loaded data if successful, None otherwise
    """
    print("🔄 Loading paleoclimate data...")
    
    data = {}
    
    try:
        # Loading data from the first proxy
        proxy1_data = pd.read_csv(proxy1_file)
        # Detecting column names automatically
        original_cols1 = proxy1_data.columns.tolist()
        proxy1_name = original_cols1[1] if len(original_cols1) > 1 else 'proxy1'
        
        # Standardizing column names
        proxy1_data.columns = ['age_kyr', 'proxy1_values']
        print(f"✅ Data {proxy1_name} loaded: {len(proxy1_data)} points")
        
        data['proxy1_data'] = proxy1_data
        data['proxy1_name'] = proxy1_name
        data['proxy1_units'] = ''  # Can be expanded to detect units
        
    except Exception as e:
        print(f"❌ Error loading first proxy: {e}")
        return False, None
        
    try:
        # Loading data from the second proxy
        proxy2_data = pd.read_csv(proxy2_file)
        # Detecting column names automatically
        original_cols2 = proxy2_data.columns.tolist()
        proxy2_name = original_cols2[1] if len(original_cols2) > 1 else 'proxy2'
        
        # Standardizing column names
        proxy2_data.columns = ['age_kyr', 'proxy2_values']
        print(f"✅ Data {proxy2_name} loaded: {len(proxy2_data)} points")
        
        data['proxy2_data'] = proxy2_data
        data['proxy2_name'] = proxy2_name
        data['proxy2_units'] = ''  # Can be expanded to detect units
        
    except Exception as e:
        print(f"❌ Error loading second proxy: {e}")
        return False, None
    
    # Clean and sort data
    data['proxy1_data'] = data['proxy1_data'].dropna().sort_values('age_kyr').reset_index(drop=True)
    data['proxy2_data'] = data['proxy2_data'].dropna().sort_values('age_kyr').reset_index(drop=True)
    
    return True, data

def interpolate_to_common_grid(proxy1_data: pd.DataFrame, proxy2_data: pd.DataFrame, 
                                resolution: float = config.INTERPOLATION_RESOLUTION) -> pd.DataFrame:
    """
    Interpolate both series to a common temporal grid
    
    Parameters:
    -----------
    proxy1_data : pd.DataFrame
        First proxy data with columns 'age_kyr' and 'proxy1_values'
    proxy2_data : pd.DataFrame
        Second proxy data with columns 'age_kyr' and 'proxy2_values'
    resolution : float
        Temporal resolution in kyr (default: 1.0)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with interpolated data on common grid
    """
    print(f"\n🔄 Interpolating to common grid (resolution: {resolution} kyr)...")
    
    # Defining common temporal range (overlap)
    min_age = max(proxy1_data['age_kyr'].min(), proxy2_data['age_kyr'].min())
    max_age = min(proxy1_data['age_kyr'].max(), proxy2_data['age_kyr'].max())
    
    # Creating uniform temporal grid
    common_ages = np.arange(min_age, max_age + resolution, resolution)
    
    # Linear interpolation for 1st proxy
    interp_proxy1 = interp1d(
        proxy1_data['age_kyr'], 
        proxy1_data['proxy1_values'],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Linear interpolation for the 2nd proxy
    interp_proxy2 = interp1d(
        proxy2_data['age_kyr'],
        proxy2_data['proxy2_values'],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Creating DF with interpolated data
    interpolated_data = pd.DataFrame({
        'age_kyr': common_ages,
        'proxy1_values': interp_proxy1(common_ages),
        'proxy2_values': interp_proxy2(common_ages)
    }).dropna().reset_index(drop=True)
    
    print(f"✅ Interpolation completed: {len(interpolated_data)} points")
    print(f"   Final range: {interpolated_data['age_kyr'].min():.1f} - {interpolated_data['age_kyr'].max():.1f} kyr")
    
    return interpolated_data

def export_rolling_window_results(rolling_correlation: pd.DataFrame, proxy1_name: str, proxy2_name: str,
                                    experiment_dir: str, filename: str = 'rolling_correlation_results.csv') -> None:
    """
    Export rolling window correlation results to CSV file
    
    Parameters:
    -----------
    rolling_correlation : pd.DataFrame
        Rolling correlation results
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    experiment_dir : str
        Experiment directory path
    filename : str
        Name of the file to export
    """
    # Create full path with experiment directory
    full_path = f'{experiment_dir}/{filename}'
        
    # Create export DataFrame with original column names
    export_data = rolling_correlation.copy()
    export_data = export_data.rename(columns={
        'proxy1_values': proxy1_name,
        'proxy2_values': proxy2_name
    })
    
    export_data.to_csv(full_path, index=False, sep=';', decimal=',')
    print(f"✅ Results exported to: {full_path}")

def export_spectral_results(cwt1_power: np.ndarray, cwt2_power: np.ndarray, coherence: np.ndarray,
                            periods: np.ndarray, freqs: np.ndarray, proxy1_name: str, proxy2_name: str,
                            experiment_dir: str, filename: str = 'spectral_analysis_results.csv') -> None:
    """
    Export spectral analysis results to CSV file
    
    Parameters:
    -----------
    cwt1_power : np.ndarray
        CWT power for proxy 1
    cwt2_power : np.ndarray
        CWT power for proxy 2
    coherence : np.ndarray
        Wavelet coherence
    periods : np.ndarray
        Periods array
    freqs : np.ndarray
        Frequencies array
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    experiment_dir : str
        Experiment directory path
    filename : str
        Name of the file to export
    """
    # Create full path with experiment directory
    full_path = f'{experiment_dir}/{filename}'
    
    # Calculate global spectra and coherence
    global_power1 = np.var(cwt1_power, axis=1)
    global_power2 = np.var(cwt2_power, axis=1)
    global_coherence = np.mean(coherence, axis=1)
    
    # Create export DataFrame
    export_data = pd.DataFrame({
        'period_kyr': periods.round(3),
        'frequency_1_per_kyr': freqs.round(3),
        f'{proxy1_name}_global_power': global_power1.round(3),
        f'{proxy2_name}_global_power': global_power2.round(3),
        'global_coherence': global_coherence.round(3),
        'significant_coherence': (global_coherence > config.SPECTRAL_ANALYSIS['coherence_threshold']).astype(int)
    })
    
    export_data.to_csv(full_path, index=False, sep=';', decimal=',')
    print(f"✅ Spectral results exported to: {full_path}")

def export_leadlag_results(leadlag_results: Dict[str, Any], experiment_dir: str, 
                            filename: str = 'leadlag_analysis_results.csv') -> None:
    """
    Export lead-lag analysis results to CSV file
    
    Parameters:
    -----------
    leadlag_results : dict
        Lead-lag analysis results
    experiment_dir : str
        Experiment directory path
    filename : str
        Name of the file to export
    """
    # Create full path with experiment directory
    full_path = f'{experiment_dir}/{filename}'
    
    # Export cross-correlation results if available
    if 'cross_correlation' in leadlag_results['detailed_results']:
        cross_corr_data = leadlag_results['detailed_results']['cross_correlation']
        
        # Create export DataFrame
        export_data = []
        for corr_type, data in cross_corr_data.items():
            lags = data['lags_kyr']
            correlations = data['correlations']
            
            for lag, corr in zip(lags, correlations):
                export_data.append({
                    'lag_kyr': lag,
                    'correlation_type': corr_type,
                    'correlation_value': round(float(corr), 3)
                })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(full_path, index=False, sep=';', decimal=',')
        print(f"✅ Results exported to: {full_path}")
    else:
        print("⚠️ No cross-correlation data available for export")