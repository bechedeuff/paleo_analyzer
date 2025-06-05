from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import bootstrap
import warnings
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Import configurations
from . import configurations as config
from .utils import load_paleoclimate_data

warnings.filterwarnings('ignore')

# Apply matplotlib and seaborn styling from configurations
plt.style.use(config.MATPLOTLIB_STYLE)
sns.set_palette(config.SEABORN_PALETTE)
plt.rcParams['figure.figsize'] = config.DEFAULT_FIGURE_SIZE
plt.rcParams['font.size'] = config.DEFAULT_FONT_SIZE

class PaleoclimateLead_LagAnalyzer:
    """
    Lead-Lag Analysis for Paleoclimate Time Series
    
    This class implements multiple methods for lead-lag analysis:
    1. Standard cross-correlation
    2. Cross-correlation AUC (Area Under Curve) method
    3. Cross-correlation at maximum lag
    
    Features:
    - Multiple correlation types (Pearson, Spearman, Kendall)
    - Bootstrap confidence intervals
    - Data preprocessing (detrending, normalization)
    - Visualizations
    """
    
    def __init__(self, experiment_dir: Optional[str] = None) -> None:
        self.proxy1_data: Optional[pd.DataFrame] = None
        self.proxy2_data: Optional[pd.DataFrame] = None
        self.proxy1_name: Optional[str] = None
        self.proxy2_name: Optional[str] = None
        self.proxy1_units: Optional[str] = None
        self.proxy2_units: Optional[str] = None
        self.interpolated_data: Optional[pd.DataFrame] = None
        self.leadlag_results: Optional[Dict[str, Any]] = None
        self.experiment_dir: str = experiment_dir or 'results'
        
    def load_data(self, proxy1_file: str, proxy2_file: str) -> bool:
        """
        Load paleoclimate data from CSV files using common utility function
        
        Parameters:
        -----------
        proxy1_file : str
            Path to CSV file of the first proxy
        proxy2_file : str
            Path to CSV file of the second proxy
            
        Returns:
        --------
        bool : True if loading successful, False otherwise
        """
        success, data = load_paleoclimate_data(proxy1_file, proxy2_file)
        
        if success and data:
            # Set instance attributes from loaded data
            self.proxy1_data = data['proxy1_data']
            self.proxy2_data = data['proxy2_data']
            self.proxy1_name = data['proxy1_name']
            self.proxy2_name = data['proxy2_name']
            self.proxy1_units = data['proxy1_units']
            self.proxy2_units = data['proxy2_units']
            return True
        else:
            return False
            
    def interpolate_to_common_grid(self, resolution: float = config.INTERPOLATION_RESOLUTION) -> None:
        """
        Interpolate both series to a common temporal grid
        
        Parameters:
        -----------
        resolution : float
            Temporal resolution in kyr (default: 1.0)
        """
        print(f"\n🔄 Interpolating to common grid (resolution: {resolution} kyr)...")
        
        # Defining common temporal range (overlap)
        min_age = max(self.proxy1_data['age_kyr'].min(), self.proxy2_data['age_kyr'].min())
        max_age = min(self.proxy1_data['age_kyr'].max(), self.proxy2_data['age_kyr'].max())
        
        # Creating uniform temporal grid
        common_ages = np.arange(min_age, max_age + resolution, resolution)
        
        # Linear interpolation for both proxies
        interp_proxy1 = interp1d(
            self.proxy1_data['age_kyr'], 
            self.proxy1_data['proxy1_values'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        interp_proxy2 = interp1d(
            self.proxy2_data['age_kyr'],
            self.proxy2_data['proxy2_values'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Creating DataFrame with interpolated data
        self.interpolated_data = pd.DataFrame({
            'age_kyr': common_ages,
            'proxy1_values': interp_proxy1(common_ages),
            'proxy2_values': interp_proxy2(common_ages)
        }).dropna().reset_index(drop=True)
        
        print(f"✅ Interpolation completed: {len(self.interpolated_data)} points")
        print(f"   Final range: {self.interpolated_data['age_kyr'].min():.1f} - {self.interpolated_data['age_kyr'].max():.1f} kyr")
        
    def preprocess_data(self, detrend: bool = config.LEADLAG_ANALYSIS['detrend_data'],
                        normalize: bool = config.LEADLAG_ANALYSIS['normalize_data']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data with optional detrending and normalization
        
        Parameters:
        -----------
        detrend : bool
            Whether to detrend the data
        normalize : bool
            Whether to normalize (z-score) the data
            
        Returns:
        --------
        tuple : Preprocessed proxy1 and proxy2 arrays
        """
        if self.interpolated_data is None:
            raise ValueError("Execute interpolate_to_common_grid() first!")
            
        proxy1_values = self.interpolated_data['proxy1_values'].values.copy()
        proxy2_values = self.interpolated_data['proxy2_values'].values.copy()
        
        # Detrending (remove linear trend)
        if detrend:
            time_index = np.arange(len(proxy1_values))
            
            # Detrend proxy1
            p1_coeffs = np.polyfit(time_index, proxy1_values, 1)
            p1_trend = np.polyval(p1_coeffs, time_index)
            proxy1_values = proxy1_values - p1_trend
            
            # Detrend proxy2
            p2_coeffs = np.polyfit(time_index, proxy2_values, 1)
            p2_trend = np.polyval(p2_coeffs, time_index)
            proxy2_values = proxy2_values - p2_trend
            
        # Normalization (z-score)
        if normalize:
            proxy1_values = (proxy1_values - np.mean(proxy1_values)) / np.std(proxy1_values)
            proxy2_values = (proxy2_values - np.mean(proxy2_values)) / np.std(proxy2_values)
            
        return proxy1_values, proxy2_values
    
    def cross_correlation_at_lag(self, proxy1: np.ndarray, proxy2: np.ndarray, 
                                    lag: int, correlation_method: str = 'pearson') -> float:
        """
        Compute cross-correlation for a given lag
        
        Parameters:
        -----------
        proxy1, proxy2 : np.ndarray
            Time series data
        lag : int
            Lag in time steps
        correlation_method : str
            Type of correlation ('pearson', 'spearman', 'kendall')
            
        Returns:
        --------
        float : Cross-correlation value
        """
        if lag == 0:
            # No lag
            data = np.column_stack([proxy1, proxy2])
            data = pd.DataFrame(data)
        elif lag > 0:
            # Positive lag: proxy1 leads proxy2
            data = np.column_stack([proxy1[:-lag], proxy2[lag:]])
            data = pd.DataFrame(data)
        else:
            # Negative lag: proxy2 leads proxy1
            lag = abs(lag)
            data = np.column_stack([proxy1[lag:], proxy2[:-lag]])
            data = pd.DataFrame(data)
            
        # Remove any remaining NaN values
        data = data.dropna()
        
        if len(data) < 10:  # Minimum data points
            return np.nan
            
        # Calculate correlation
        if correlation_method in ['pearson', 'spearman', 'kendall']:
            corr_matrix = data.corr(method=correlation_method)
            return corr_matrix.iloc[0, 1]
        else:
            raise ValueError(f"Unsupported correlation method: {correlation_method}")
    
    def ccf_auc_method(self, proxy1: np.ndarray, proxy2: np.ndarray, 
                        max_lag: int, correlation_method: str = 'pearson') -> float:
        """
        Cross-correlation AUC method
        
        Parameters:
        -----------
        proxy1, proxy2 : np.ndarray
            Time series data
        max_lag : int
            Maximum lag to test
        correlation_method : str
            Type of correlation
            
        Returns:
        --------
        float : Lead-lag measure
        """
        lags = np.arange(1, max_lag + 1)
        lags = np.r_[-lags, lags]
        
        correlations = {}
        for lag in lags:
            correlations[lag] = self.cross_correlation_at_lag(proxy1, proxy2, lag, correlation_method)
            
        correlations = pd.Series(correlations).dropna()
        
        # Calculate areas
        A = correlations[correlations.index > 0]  # Positive lags (proxy1 leads)
        A = np.abs(A).sum()
        
        B = correlations[correlations.index < 0]  # Negative lags (proxy2 leads)
        B = np.abs(B).sum()
        
        if A + B == 0:
            return 0.0
            
        # Lead-lag measure: normalized difference
        return (A - B) / (A + B)
    
    def ccf_at_max_lag_method(self, proxy1: np.ndarray, proxy2: np.ndarray,
                                max_lag: int, correlation_method: str = 'pearson') -> Tuple[float, int]:
        """
        Cross-correlation at maximum lag method
        
        Parameters:
        -----------
        proxy1, proxy2 : np.ndarray
            Time series data
        max_lag : int
            Maximum lag to test
        correlation_method : str
            Type of correlation
            
        Returns:
        --------
        tuple : (lead_lag_measure, optimal_lag)
        """
        lags = np.arange(-max_lag, max_lag + 1)
        correlations = {}
        
        for lag in lags:
            correlations[lag] = self.cross_correlation_at_lag(proxy1, proxy2, lag, correlation_method)
            
        correlations = pd.Series(correlations).dropna()
        
        if len(correlations) == 0:
            return 0.0, 0
            
        # Find maximum absolute correlation
        abs_corr = correlations.abs()
        optimal_lag = abs_corr.idxmax()
        max_correlation = correlations[optimal_lag]
        
        return max_correlation, optimal_lag
    
    def bootstrap_confidence_intervals(self, proxy1: np.ndarray, proxy2: np.ndarray,
                                        method: str, n_bootstrap: int = config.LEADLAG_ANALYSIS['bootstrap_iterations'],
                                        confidence_level: float = config.LEADLAG_ANALYSIS['confidence_level'],
                                       **kwargs) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for lead-lag measures
        
        Parameters:
        -----------
        proxy1, proxy2 : np.ndarray
            Time series data
        method : str
            Method to use ('ccf_auc', 'ccf_at_max_lag')
        n_bootstrap : int
            Number of bootstrap iterations
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        **kwargs : dict
            Additional arguments for the method
            
        Returns:
        --------
        tuple : (lower_bound, upper_bound)
        """
        def statistic(paired_data, axis=-1):
            """
            Statistic function for scipy's bootstrap function
            paired_data shape: (n_samples, 2) where each row is [proxy1_val, proxy2_val]
            """
            try:
                # Extract the two time series from paired data
                p1_1d = paired_data[:, 0]
                p2_1d = paired_data[:, 1]
                
                if method == 'ccf_auc':
                    result = self.ccf_auc_method(p1_1d, p2_1d, **kwargs)
                    return np.array([result])  # Return as 1D array
                elif method == 'ccf_at_max_lag':
                    result, _ = self.ccf_at_max_lag_method(p1_1d, p2_1d, **kwargs)
                    return np.array([result])  # Return as 1D array
                else:
                    raise ValueError(f"Unsupported method for bootstrap: {method}")
            except:
                return np.array([np.nan])  # Return NaN as 1D array
        
        try:
            # Prepare data as paired observations: each row is [proxy1_val, proxy2_val]
            paired_data = np.column_stack([proxy1, proxy2])  # Shape: (n_samples, 2)
            
            # Configure bootstrap
            rng = np.random.default_rng()
            
            # Run bootstrap
            result = bootstrap(
                (paired_data,),
                statistic=statistic,
                n_resamples=n_bootstrap,
                confidence_level=confidence_level,
                method='percentile',
                random_state=rng,
                axis=0  # Resample along the first axis (rows/observations)
            )
            
            # Extract confidence interval (take first element since we return 1D arrays)
            lower_bound = result.confidence_interval.low[0] if hasattr(result.confidence_interval.low, '__len__') else result.confidence_interval.low
            upper_bound = result.confidence_interval.high[0] if hasattr(result.confidence_interval.high, '__len__') else result.confidence_interval.high
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            # Fallback to manual implementation if scipy bootstrap fails
            print(f"⚠️ scipy.stats.bootstrap failed ({e}), using fallback method...")
            
            # Manual bootstrap as fallback
            data = np.column_stack([proxy1, proxy2])
            bootstrap_results = []
            
            for _ in range(min(n_bootstrap, 100)):  # Limit iterations for fallback
                try:
                    # Resample with replacement
                    n = len(data)
                    indices = np.random.choice(n, n, replace=True)
                    p1_boot = data[indices, 0]
                    p2_boot = data[indices, 1]
                    
                    if method == 'ccf_auc':
                        result = self.ccf_auc_method(p1_boot, p2_boot, **kwargs)
                    elif method == 'ccf_at_max_lag':
                        result, _ = self.ccf_at_max_lag_method(p1_boot, p2_boot, **kwargs)
                    else:
                        continue
                        
                    if not np.isnan(result):
                        bootstrap_results.append(result)
                except:
                    continue
                    
            if len(bootstrap_results) < 5:
                return np.nan, np.nan
                
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_results, lower_percentile)
            upper_bound = np.percentile(bootstrap_results, upper_percentile)
            
            return float(lower_bound), float(upper_bound)
    
    def calculate_comprehensive_leadlag_analysis(self, 
                                                max_lag_kyr: Optional[float] = None,
                                                lag_step_kyr: Optional[float] = None,
                                                methods: Optional[List[str]] = None,
                                                correlation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive lead-lag analysis using multiple methods
        
        Parameters:
        -----------
        max_lag_kyr : float, optional
            Maximum lag to test (kyr). If None, uses config
        lag_step_kyr : float, optional
            Step size for lag testing (kyr). If None, uses config
        methods : List[str], optional
            Methods to use. If None, uses config
        correlation_types : List[str], optional
            Correlation types to test. If None, uses config
            
        Returns:
        --------
        dict : Comprehensive lead-lag analysis results
        """
        if self.interpolated_data is None:
            raise ValueError("Execute interpolate_to_common_grid() first!")
            
        print("\n🔄 Performing comprehensive lead-lag analysis...")
        
        # Use configuration values if not provided
        if max_lag_kyr is None:
            max_lag_kyr = config.LEADLAG_ANALYSIS['max_lag_kyr']
        if lag_step_kyr is None:
            lag_step_kyr = config.LEADLAG_ANALYSIS['lag_step_kyr']
        if methods is None:
            methods = config.LEADLAG_ANALYSIS['methods']
        if correlation_types is None:
            correlation_types = config.LEADLAG_ANALYSIS['correlation_types']
            
        # Preprocess data
        proxy1_processed, proxy2_processed = self.preprocess_data()
        
        # Convert lag parameters to time steps
        resolution = self.interpolated_data['age_kyr'].iloc[1] - self.interpolated_data['age_kyr'].iloc[0]
        max_lag_steps = int(max_lag_kyr / resolution)
        lag_step_steps = max(1, int(lag_step_kyr / resolution))
        
        results = {
            'analysis_info': {
                'max_lag_kyr': max_lag_kyr,
                'lag_step_kyr': lag_step_kyr,
                'max_lag_steps': max_lag_steps,
                'lag_step_steps': lag_step_steps,
                'resolution_kyr': resolution,
                'methods': methods,
                'correlation_types': correlation_types,
                'preprocessing': {
                    'detrend': config.LEADLAG_ANALYSIS['detrend_data'],
                    'normalize': config.LEADLAG_ANALYSIS['normalize_data']
                }
            },
            'detailed_results': {},
            'summary': {}
        }
        
        # Standard cross-correlation analysis
        if 'cross_correlation' in methods:
            print("   • Standard cross-correlation analysis...")
            lags_kyr = np.arange(-max_lag_kyr, max_lag_kyr + lag_step_kyr, lag_step_kyr)
            lags_steps = np.arange(-max_lag_steps, max_lag_steps + lag_step_steps, lag_step_steps)
            
            cross_corr_results = {}
            for corr_type in correlation_types:
                correlations = []
                for lag_step in lags_steps:
                    corr = self.cross_correlation_at_lag(proxy1_processed, proxy2_processed, 
                                                        lag_step, corr_type)
                    correlations.append(corr)
                    
                correlations = np.array(correlations)
                
                # Find optimal lag
                valid_mask = ~np.isnan(correlations)
                if np.any(valid_mask):
                    valid_corr = correlations[valid_mask]
                    valid_lags = lags_kyr[valid_mask]
                    
                    optimal_idx = np.argmax(np.abs(valid_corr))
                    optimal_lag_kyr = valid_lags[optimal_idx]
                    optimal_correlation = valid_corr[optimal_idx]
                else:
                    optimal_lag_kyr = 0.0
                    optimal_correlation = 0.0
                
                cross_corr_results[corr_type] = {
                    'lags_kyr': lags_kyr,
                    'correlations': correlations,
                    'optimal_lag_kyr': optimal_lag_kyr,
                    'optimal_correlation': optimal_correlation
                }
                
            results['detailed_results']['cross_correlation'] = cross_corr_results
        
        # CCF AUC method
        if 'ccf_auc' in methods:
            print("   • CCF AUC method...")
            ccf_auc_results = {}
            for corr_type in correlation_types:
                auc_measure = self.ccf_auc_method(proxy1_processed, proxy2_processed, 
                                                 max_lag_steps, corr_type)
                
                # Bootstrap confidence intervals
                lower_ci, upper_ci = self.bootstrap_confidence_intervals(
                    proxy1_processed, proxy2_processed, 'ccf_auc',
                    max_lag=max_lag_steps, correlation_method=corr_type
                )
                
                ccf_auc_results[corr_type] = {
                    'auc_measure': round(auc_measure, 3),
                    'confidence_interval': (lower_ci, upper_ci)
                }
                
            results['detailed_results']['ccf_auc'] = ccf_auc_results
        
        # CCF at max lag method
        if 'ccf_at_max_lag' in methods:
            print("   • CCF at maximum lag method...")
            ccf_max_results = {}
            for corr_type in correlation_types:
                max_corr, optimal_lag_steps = self.ccf_at_max_lag_method(
                    proxy1_processed, proxy2_processed, max_lag_steps, corr_type
                )
                optimal_lag_kyr = optimal_lag_steps * resolution
                
                # Bootstrap confidence intervals
                lower_ci, upper_ci = self.bootstrap_confidence_intervals(
                    proxy1_processed, proxy2_processed, 'ccf_at_max_lag',
                    max_lag=max_lag_steps, correlation_method=corr_type
                )
                
                ccf_max_results[corr_type] = {
                    'max_correlation': round(float(max_corr), 3),
                    'optimal_lag_kyr': optimal_lag_kyr,
                    'optimal_lag_steps': optimal_lag_steps,
                    'confidence_interval': (lower_ci, upper_ci)
                }
                
            results['detailed_results']['ccf_at_max_lag'] = ccf_max_results
        
        # Create summary
        print("   • Creating summary...")
        summary = {}
        for method in methods:
            if method in results['detailed_results']:
                summary[method] = {}
                for corr_type in correlation_types:
                    method_data = results['detailed_results'][method][corr_type]
                    
                    if method == 'cross_correlation':
                        summary[method][corr_type] = {
                            'optimal_lag_kyr': method_data['optimal_lag_kyr'],
                            'optimal_correlation': method_data['optimal_correlation']
                        }
                    elif method == 'ccf_auc':
                        summary[method][corr_type] = {
                            'auc_measure': method_data['auc_measure'],
                            'interpretation': self._interpret_auc_measure(method_data['auc_measure'])
                        }
                    elif method == 'ccf_at_max_lag':
                        summary[method][corr_type] = {
                            'optimal_lag_kyr': method_data['optimal_lag_kyr'],
                            'max_correlation': method_data['max_correlation']
                        }
        
        results['summary'] = summary
        
        # Store results
        self.leadlag_results = results
        
        print("✅ Lead-lag analysis completed!")
        return results
    
    def _interpret_auc_measure(self, auc_value: float) -> str:
        """
        Interpret AUC measure value
        
        Parameters:
        -----------
        auc_value : float
            AUC measure value
            
        Returns:
        --------
        str : Interpretation
        """
        if auc_value > 0.1:
            return f"{self.proxy1_name} leads {self.proxy2_name} (strong)"
        elif auc_value > 0.05:
            return f"{self.proxy1_name} leads {self.proxy2_name} (moderate)"
        elif auc_value > 0.01:
            return f"{self.proxy1_name} leads {self.proxy2_name} (weak)"
        elif auc_value < -0.1:
            return f"{self.proxy2_name} leads {self.proxy1_name} (strong)"
        elif auc_value < -0.05:
            return f"{self.proxy2_name} leads {self.proxy1_name} (moderate)"
        elif auc_value < -0.01:
            return f"{self.proxy2_name} leads {self.proxy1_name} (weak)"
        else:
            return "Synchronous or no clear relationship"
    
    def plot_comprehensive_leadlag_analysis(self, figsize: Tuple[int, int] = None) -> None:
        """
        Create comprehensive lead-lag analysis plot
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size. If None, uses config
        """
        if self.leadlag_results is None:
            raise ValueError("Execute calculate_comprehensive_leadlag_analysis() first!")
            
        if figsize is None:
            figsize = config.LEADLAG_PLOTS['comprehensive_figsize']
            
        # Set up figure with 2 panels: top spanning, middle spanning, bottom spanning
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        # =============================================================================
        # Panel A: Time Series Data (Top, spanning full width)
        # =============================================================================
        ax1 = fig.add_subplot(gs[0])
        ax1_twin = ax1.twinx()
        
        # Determine leader/lagger based on optimal lag from cross-correlation
        optimal_lag = 0
        if 'cross_correlation' in self.leadlag_results['detailed_results']:
            default_corr_type = config.LEADLAG_PLOTS['default_correlation_type']
            if default_corr_type in self.leadlag_results['detailed_results']['cross_correlation']:
                optimal_lag = self.leadlag_results['detailed_results']['cross_correlation'][default_corr_type]['optimal_lag_kyr']
        
        # Assign leader/lagger labels based on optimal lag
        if optimal_lag > 0:
            proxy1_label = f'{self.proxy1_name} (Leader)'
            proxy2_label = f'{self.proxy2_name} (Lagger)'
        elif optimal_lag < 0:
            proxy1_label = f'{self.proxy1_name} (Lagger)'
            proxy2_label = f'{self.proxy2_name} (Leader)'
        else:
            proxy1_label = f'{self.proxy1_name}'
            proxy2_label = f'{self.proxy2_name}'
        
        line1 = ax1.plot(self.interpolated_data['age_kyr'], 
                        self.interpolated_data['proxy1_values'], 
                        'b-', linewidth=2, alpha=0.8, label=proxy1_label)
        line2 = ax1_twin.plot(self.interpolated_data['age_kyr'], 
                             self.interpolated_data['proxy2_values'], 
                             'r-', linewidth=2, alpha=0.8, label=proxy2_label)
        
        ax1.set_xlabel('Age (kyr ago)')
        ax1.set_ylabel(f'{self.proxy1_name}', color='blue')
        ax1_twin.set_ylabel(f'{self.proxy2_name}', color='red')
        ax1.set_title('A) Time Series Data', fontweight='bold')
        ax1.grid(True, alpha=0.1)
        ax1.invert_xaxis()
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')
        
        # =============================================================================
        # Panel B: Contrast
        # This plot shows the correlation between two proxies at different lags
        # With the objective to determine the direction of the correlation and which proxy is leading
        # Negative values mean that the proxy2 leads the proxy1
        # Positive values mean that the proxy1 leads the proxy2
        # =============================================================================
        ax2 = fig.add_subplot(gs[1])
        
        if 'cross_correlation' in self.leadlag_results['detailed_results']:
            contrast_color = config.LEADLAG_PLOTS['contrast_line_color']
            default_corr_type = config.LEADLAG_PLOTS['default_correlation_type']
            
            if default_corr_type in self.leadlag_results['detailed_results']['cross_correlation']:
                ccf_data = self.leadlag_results['detailed_results']['cross_correlation'][default_corr_type]
                lags_kyr = ccf_data['lags_kyr']
                correlations = ccf_data['correlations']
                optimal_lag = ccf_data['optimal_lag_kyr']
                optimal_corr = ccf_data['optimal_correlation']
                
                # Plot cross-correlation function
                ax2.plot(lags_kyr, correlations, 'b-', color=contrast_color, linewidth=2.5, alpha=0.9)
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                # Highlight optimal lag
                ax2.plot(optimal_lag, optimal_corr, 'ro', color="black", markersize=10, label=f'Optimal lag: {optimal_lag:.1f} kyr')
                
                # Add interpretation
                interpretation_text = f'Positive means {self.proxy1_name} leads {self.proxy2_name} \nNegative means {self.proxy2_name} leads {self.proxy1_name}'
                
                ax2.set_xlabel(f'Lag (kyr) \n{interpretation_text}')
                ax2.set_ylabel('Cross-Correlation')
                ax2.set_title('B) Contrast', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
        
        plt.suptitle(f'Lead-Lag Analysis: {self.proxy1_name} vs {self.proxy2_name}', 
                        fontsize=16, fontweight='bold')
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/leadlag_analysis.png'
        plt.savefig(filename, dpi=config.LEADLAG_PLOTS['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✅ Lead-lag plot saved: {filename}")
    
    def export_results(self, filename: str = 'leadlag_analysis_results.csv') -> None:
        """
        Export results to CSV file
        
        Parameters:
        -----------
        filename : str
            Name of the file to export
        """
        if self.leadlag_results is None:
            raise ValueError("Execute calculate_comprehensive_leadlag_analysis() first!")
            
        # Create full path with experiment directory
        full_path = f'{self.experiment_dir}/{filename}'
        
        # Export cross-correlation results if available
        if 'cross_correlation' in self.leadlag_results['detailed_results']:
            cross_corr_data = self.leadlag_results['detailed_results']['cross_correlation']
            
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
    
    def create_json_metadata(self) -> str:
        """
        Create a JSON file with lead-lag analysis metadata
        
        Returns:
        --------
        str
            Path to the generated JSON file
        """
        if self.leadlag_results is None:
            raise ValueError("Execute calculate_comprehensive_leadlag_analysis() first!")
            
        # Prepare JSON-serializable data
        metadata = {
            "analysis_info": {
                "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "analysis_type": "lead_lag",
                "proxy1_name": self.proxy1_name,
                "proxy2_name": self.proxy2_name
            },
            "data_summary": {
                "total_points": len(self.interpolated_data),
                "age_range_kyr": {
                    "min": float(self.interpolated_data['age_kyr'].min()),
                    "max": float(self.interpolated_data['age_kyr'].max())
                }
            },
            "analysis_parameters": self.leadlag_results['analysis_info'],
            "summary_results": {},
            "configuration_used": {
                key: config.LEADLAG_ANALYSIS[key] 
                for key in config.LEADLAG_ANALYSIS 
                if isinstance(config.LEADLAG_ANALYSIS[key], (int, float, bool, str, list))
            }
        }
        
        # Add summary results (converting numpy types to Python types)
        for method, method_data in self.leadlag_results['summary'].items():
            metadata["summary_results"][method] = {}
            for corr_type, results in method_data.items():
                metadata["summary_results"][method][corr_type] = {}
                for key, value in results.items():
                    if isinstance(value, (np.integer, np.floating)):
                        metadata["summary_results"][method][corr_type][key] = round(float(value), 3)
                    else:
                        metadata["summary_results"][method][corr_type][key] = value
        
        # Save JSON
        json_file = f'{self.experiment_dir}/leadlag_analysis_metadata.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return json_file
    
    def create_pdf_report(self) -> str:
        """
        Create a well-formatted PDF report
        
        Returns:
        --------
        str
            Path to the generated PDF file
        """
        if self.leadlag_results is None:
            raise ValueError("Execute calculate_comprehensive_leadlag_analysis() first!")
            
        # PDF configuration
        pdf_file = f'{self.experiment_dir}/leadlag_analysis_report.pdf'
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,
            textColor=colors.black
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.black
        )
        
        # Title
        story.append(Paragraph("LEAD-LAG ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # General information
        analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        info_data = [
            ['Analysis date:', analysis_date],
            ['Proxy 1:', self.proxy1_name],
            ['Proxy 2:', self.proxy2_name],
            ['Analysis methods:', ', '.join(self.leadlag_results['analysis_info']['methods'])],
            ['Correlation types:', ', '.join(self.leadlag_results['analysis_info']['correlation_types'])],
            ['Max lag tested:', f"{self.leadlag_results['analysis_info']['max_lag_kyr']} kyr"],
            ['Total data points:', str(len(self.interpolated_data))]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Results summary
        story.append(Paragraph("RESULTS SUMMARY", section_style))
        
        for method, method_data in self.leadlag_results['summary'].items():
            story.append(Paragraph(f"{method.replace('_', ' ').title()}", section_style))
            
            results_data = [['Correlation Type', 'Key Result', 'Value']]
            for corr_type, results in method_data.items():
                if method == 'cross_correlation':
                    results_data.append([
                        corr_type.capitalize(),
                        'Optimal lag (kyr)',
                        f"{results['optimal_lag_kyr']:.2f}"
                    ])
                    results_data.append([
                        '',
                        'Max correlation',
                        f"{results['optimal_correlation']:.3f}"
                    ])
                elif method == 'ccf_auc':
                    results_data.append([
                        corr_type.capitalize(),
                        'AUC measure',
                        f"{results['auc_measure']:.3f}"
                    ])
                    results_data.append([
                        '',
                        'Interpretation',
                        results['interpretation']
                    ])
                elif method == 'ccf_at_max_lag':
                    results_data.append([
                        corr_type.capitalize(),
                        'Optimal lag (kyr)',
                        f"{results['optimal_lag_kyr']:.2f}"
                    ])
                    results_data.append([
                        '',
                        'Max correlation',
                        f"{results['max_correlation']:.3f}"
                    ])
            
            results_table = Table(results_data, colWidths=[1.5*inch, 2*inch, 2*inch])
            results_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue)
            ]))
            story.append(results_table)
            story.append(Spacer(1, 15))
        
        # Methodology explanation
        story.append(Paragraph("METHODOLOGY", section_style))
        methodology_text = """
        <b>• Cross-correlation:</b> Standard cross-correlation function computed at various lags<br/><br/>
        <b>• CCF AUC:</b> Area under curve method comparing positive vs negative lag regions<br/><br/>
        <b>• CCF at Max Lag:</b> Cross-correlation value at the lag with maximum absolute correlation<br/><br/>
        <b>• Correlation Types:</b> Pearson (linear), Spearman (rank), Kendall (tau)
        """
        story.append(Paragraph(methodology_text, styles['Normal']))
        
        # Generate PDF
        doc.build(story)
        return pdf_file 