import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple, Optional

from src import config

def load_paleoclimate_data(proxy1_file: str, proxy2_file: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Load and clean data from CSV files
    
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