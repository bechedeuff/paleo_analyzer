import numpy as np
import pandas as pd
from typing import Dict, Any

from src import config

def export_rolling_window_results(rolling_correlation: pd.DataFrame, proxy1_name: str, proxy2_name: str,
                                    experiment_dir: str, filename: str = 'rolling_correlation_results.csv') -> None:
    """
    Export rolling window results
    
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
    Export spectral analysis results
    
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
    Export lead-lag analysis results
    
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