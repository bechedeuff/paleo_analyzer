import os
from typing import List

from src import config

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
