import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

from . import configurations as config

def create_rolling_window_metadata(rolling_correlation: pd.DataFrame, proxy1_data: pd.DataFrame, 
                                    proxy2_data: pd.DataFrame, interpolated_data: pd.DataFrame,
                                    proxy1_name: str, proxy2_name: str, window_size: int, 
                                    periods: Dict[str, pd.DataFrame], experiment_dir: str) -> str:
    """
    Create a JSON file with rolling window analysis metadata
    
    Parameters:
    -----------
    rolling_correlation : pd.DataFrame
        Rolling correlation results
    proxy1_data : pd.DataFrame
        Original proxy 1 data
    proxy2_data : pd.DataFrame
        Original proxy 2 data
    interpolated_data : pd.DataFrame
        Interpolated data
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    window_size : int
        Window size used in analysis
    periods : dict
        Dictionary with classified periods
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated JSON file
    """
    corr_stats = rolling_correlation['rolling_correlation']
    
    # Structured data
    metadata = {
        "analysis_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "proxy1_name": proxy1_name,
            "proxy2_name": proxy2_name,
            "window_size_kyr": window_size,
            "analysis_period": {
                "start_kyr": float(interpolated_data['age_kyr'].min()),
                "end_kyr": float(interpolated_data['age_kyr'].max()),
                "duration_kyr": float(interpolated_data['age_kyr'].max() - interpolated_data['age_kyr'].min())
            },
            "total_points": len(rolling_correlation)
        },
        "data_summary": {
            "proxy1": {
                "name": proxy1_name,
                "original_points": len(proxy1_data),
                "age_range": {
                    "min_kyr": float(proxy1_data['age_kyr'].min()),
                    "max_kyr": float(proxy1_data['age_kyr'].max())
                },
                "value_range": {
                    "min": float(proxy1_data['proxy1_values'].min()),
                    "max": float(proxy1_data['proxy1_values'].max())
                }
            },
            "proxy2": {
                "name": proxy2_name,
                "original_points": len(proxy2_data),
                "age_range": {
                    "min_kyr": float(proxy2_data['age_kyr'].min()),
                    "max_kyr": float(proxy2_data['age_kyr'].max())
                },
                "value_range": {
                    "min": float(proxy2_data['proxy2_values'].min()),
                    "max": float(proxy2_data['proxy2_values'].max())
                }
            }
        },
        "correlation_statistics": {
            "mean": round(float(corr_stats.mean()), 3),
            "median": round(float(corr_stats.median()), 3),
            "std": round(float(corr_stats.std()), 3),
            "min": round(float(corr_stats.min()), 3),
            "max": round(float(corr_stats.max()), 3),
            "quantiles": {
                "q25": round(float(corr_stats.quantile(0.25)), 3),
                "q75": round(float(corr_stats.quantile(0.75)), 3)
            }
        },
        "period_counts": {
            "high_positive": len(periods['high_positive']),
            "high_negative": len(periods['high_negative']),
            "decoupled": len(periods['decoupled'])
        },
        "detailed_periods": {}
    }
    
    # Detailed periods
    for category in ['high_positive', 'high_negative', 'decoupled']:
        if len(periods[category]) > 0:
            metadata["detailed_periods"][category] = []
            for _, row in periods[category].head(20).iterrows():  # Top 20 for JSON
                metadata["detailed_periods"][category].append({
                    "age_kyr": float(row['age_kyr']),
                    "correlation": float(row['rolling_correlation']),
                    "proxy1_value": float(row['proxy1_values']),
                    "proxy2_value": float(row['proxy2_values'])
                })
    
    # Save JSON
    json_file = f'{experiment_dir}/analysis_metadata.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_file

def create_spectral_metadata(interpolated_data: pd.DataFrame, cwt1_power: np.ndarray, cwt2_power: np.ndarray,
                            coherence: np.ndarray, periods: np.ndarray, proxy1_name: str, proxy2_name: str,
                            cycles_found: Dict[str, List[Tuple[float, float]]], experiment_dir: str) -> str:
    """
    Create a JSON file with spectral analysis metadata
    
    Parameters:
    -----------
    interpolated_data : pd.DataFrame
        Interpolated data
    cwt1_power : np.ndarray
        CWT power for proxy 1
    cwt2_power : np.ndarray
        CWT power for proxy 2
    coherence : np.ndarray
        Wavelet coherence
    periods : np.ndarray
        Periods array
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    cycles_found : dict
        Dictionary with identified cycles
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated JSON file
    """
    def _get_cycle_name(period: float) -> str:
        """Get the name of Milankovitch cycle based on period"""
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for target_period in cycle_periods:
                if abs(period - target_period) / target_period < 0.2:
                    return cycle_name.replace('_', ' ').title()
        return 'Unknown'
    
    # Calculate global spectra
    global_power1 = np.var(cwt1_power, axis=1)
    global_power2 = np.var(cwt2_power, axis=1)
    global_coherence = np.mean(coherence, axis=1)
    
    # Structured metadata
    metadata = {
        "analysis_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "analysis_type": "spectral_wavelet",
            "proxy1_name": proxy1_name,
            "proxy2_name": proxy2_name,
            "wavelet_type": config.SPECTRAL_ANALYSIS['wavelet_type'],
            "wavelet_parameter": config.SPECTRAL_ANALYSIS['wavelet_param'],
            "analysis_period": {
                "start_kyr": float(interpolated_data['age_kyr'].min()),
                "end_kyr": float(interpolated_data['age_kyr'].max()),
                "duration_kyr": float(interpolated_data['age_kyr'].max() - interpolated_data['age_kyr'].min())
            },
            "frequency_resolution": len(periods),
            "period_range": {
                "min_kyr": round(float(periods.min()), 3),
                "max_kyr": round(float(periods.max()), 3)
            }
        },
        "spectral_statistics": {
            "proxy1_total_power": round(float(np.sum(global_power1)), 3),
            "proxy2_total_power": round(float(np.sum(global_power2)), 3),
            "mean_coherence": round(float(np.mean(global_coherence)), 3),
            "max_coherence": round(float(np.max(global_coherence)), 3),
            "coherent_frequencies": int(np.sum(global_coherence > config.SPECTRAL_ANALYSIS['coherence_threshold']))
        },
        "milankovitch_cycles": {},
        "configuration_used": {
            "wavelet_type": config.SPECTRAL_ANALYSIS['wavelet_type'],
            "wavelet_param": config.SPECTRAL_ANALYSIS['wavelet_param'],
            "min_period": config.SPECTRAL_ANALYSIS['min_period'],
            "max_period": config.SPECTRAL_ANALYSIS['max_period'],
            "coherence_threshold": config.SPECTRAL_ANALYSIS['coherence_threshold'],
            "confidence_level": config.SPECTRAL_ANALYSIS['confidence_level']
        }
    }
    
    # Add identified cycles
    for proxy_name, cycles in cycles_found.items():
        metadata["milankovitch_cycles"][proxy_name] = []
        for period, power in cycles:
            metadata["milankovitch_cycles"][proxy_name].append({
                "cycle_type": _get_cycle_name(period),
                "period_kyr": round(float(period), 3),
                "power_coherence": round(float(power), 3)
            })
    
    # Save JSON
    json_file = f'{experiment_dir}/spectral_analysis_metadata.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_file

def create_leadlag_metadata(leadlag_results: Dict[str, Any], interpolated_data: pd.DataFrame,
                            proxy1_name: str, proxy2_name: str, experiment_dir: str) -> str:
    """
    Create a JSON file with lead-lag analysis metadata
    
    Parameters:
    -----------
    leadlag_results : dict
        Lead-lag analysis results
    interpolated_data : pd.DataFrame
        Interpolated data
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated JSON file
    """
    # Prepare JSON-serializable data
    metadata = {
        "analysis_info": {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "analysis_type": "lead_lag",
            "proxy1_name": proxy1_name,
            "proxy2_name": proxy2_name
        },
        "data_summary": {
            "total_points": len(interpolated_data),
            "age_range_kyr": {
                "min": float(interpolated_data['age_kyr'].min()),
                "max": float(interpolated_data['age_kyr'].max())
            }
        },
        "analysis_parameters": leadlag_results['analysis_info'],
        "summary_results": {},
        "configuration_used": {
            key: config.LEADLAG_ANALYSIS[key] 
            for key in config.LEADLAG_ANALYSIS 
            if isinstance(config.LEADLAG_ANALYSIS[key], (int, float, bool, str, list))
        }
    }
    
    # Add summary results (converting numpy types to Python types)
    for method, method_data in leadlag_results['summary'].items():
        metadata["summary_results"][method] = {}
        for corr_type, results in method_data.items():
            metadata["summary_results"][method][corr_type] = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata["summary_results"][method][corr_type][key] = round(float(value), 3)
                else:
                    metadata["summary_results"][method][corr_type][key] = value
    
    # Save JSON
    json_file = f'{experiment_dir}/leadlag_analysis_metadata.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_file