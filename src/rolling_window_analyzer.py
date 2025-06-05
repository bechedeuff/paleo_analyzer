from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

import warnings

# Import configurations
from . import configurations as config

warnings.filterwarnings('ignore')

# Apply matplotlib and seaborn styling from configurations
plt.style.use(config.MATPLOTLIB_STYLE)
sns.set_palette(config.SEABORN_PALETTE)
plt.rcParams['figure.figsize'] = config.DEFAULT_FIGURE_SIZE
plt.rcParams['font.size'] = config.DEFAULT_FONT_SIZE

class PaleoclimateRollingWindowAnalyzer:
    """
    Class for rolling window correlation analysis in paleoclimate data
    
    Main features:
    - Loading CSV data (any type of paleoclimate proxy)
    - Interpolation to common temporal grid
    - Rolling window correlation calculation
    - Visualization and analysis of correlation/decoupling periods
    """
    
    def __init__(self, experiment_dir: Optional[str] = None) -> None:
        self.proxy1_data: Optional[pd.DataFrame] = None
        self.proxy2_data: Optional[pd.DataFrame] = None
        self.proxy1_name: Optional[str] = None
        self.proxy2_name: Optional[str] = None
        self.proxy1_units: Optional[str] = None
        self.proxy2_units: Optional[str] = None
        self.interpolated_data: Optional[pd.DataFrame] = None
        self.rolling_correlation: Optional[pd.DataFrame] = None
        self.experiment_dir: str = experiment_dir or 'results'
        
    def calculate_rolling_correlation(self, window_size: int = config.ROLLING_WINDOW_ANALYSIS['window_size'], min_periods: Optional[int] = config.ROLLING_WINDOW_ANALYSIS['min_periods']) -> None:
        """
        Calculate rolling correlation between the two proxies
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window in points (kyr if resolution = 1)
            Default: 41 kyr (Milankovitch obliquity cycle)
        min_periods : int
            Minimum number of periods to calculate correlation
        """
        if self.interpolated_data is None:
            raise ValueError("Execute interpolate_to_common_grid() first!")
            
        if min_periods is None:
            min_periods = max(10, window_size // 2)
            
        print(f"\n🔄 Calculating rolling correlation (window: {window_size} kyr)...")
        
        # Calculating rolling correlation
        rolling_corr = self.interpolated_data['proxy1_values'].rolling(
            window=window_size,
            min_periods=min_periods
        ).corr(self.interpolated_data['proxy2_values'])
        
        # Creating DF with results
        self.rolling_correlation = pd.DataFrame({
            'age_kyr': self.interpolated_data['age_kyr'].round(3),
            'proxy1_values': self.interpolated_data['proxy1_values'].round(3),
            'proxy2_values': self.interpolated_data['proxy2_values'].round(3),
            'rolling_correlation': rolling_corr.round(3)
        }).dropna().reset_index(drop=True)
        
        print(f"✅ Rolling correlation calculated: {len(self.rolling_correlation)} points")
        
    def identify_correlation_periods(self, threshold_high: float = config.ROLLING_WINDOW_ANALYSIS['threshold_high'], threshold_low: float = config.ROLLING_WINDOW_ANALYSIS['threshold_low']) -> Dict[str, pd.DataFrame]:
        """
        Identify and return specific correlation periods
        
        Parameters:
        -----------
        threshold_high : float
            Threshold for high correlation (default: 0.7)
        threshold_low : float
            Threshold for decoupling (default: 0.2)
            
        Returns:
        --------
        dict : Dictionary with DFs of different periods
        """
        if self.rolling_correlation is None:
            raise ValueError("Execute calculate_rolling_correlation() first!")
            
        corr_data = self.rolling_correlation.copy()
        
        # Identifying periods by category
        periods = {
            'high_positive': corr_data[corr_data['rolling_correlation'] > threshold_high],
            'high_negative': corr_data[corr_data['rolling_correlation'] < -threshold_high],
            'decoupled': corr_data[abs(corr_data['rolling_correlation']) < threshold_low]
        }
        
        return periods
        
    def plot_comprehensive_analysis(self, window_size: int = config.ROLLING_WINDOW_ANALYSIS['window_size'], figsize: Tuple[int, int] = config.ROLLING_WINDOW_PLOTS['comprehensive_figsize']) -> None:
        """
        Create comprehensive visualization with 4 subplots and save to file
        
        Parameters:
        -----------
        window_size : int
            Size of the window for the title
            Default: 41 kyr (Milankovitch obliquity cycle)
        figsize : tuple
            Size of the figure
        """
        if self.rolling_correlation is None:
            raise ValueError("Execute calculate_rolling_correlation() first!")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Rolling Window Correlation Analysis - {self.proxy1_name} vs {self.proxy2_name}\n'
                        f'Window: {window_size} kyr', 
                        fontsize=16, fontweight='bold')
        
        # 1. Original time series with double axes
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(self.interpolated_data['age_kyr'], 
                        self.interpolated_data['proxy1_values'], 
                        'b-', linewidth=1.5, alpha=0.8, label=self.proxy1_name)
        line2 = ax1_twin.plot(self.interpolated_data['age_kyr'], 
                                self.interpolated_data['proxy2_values'].round(3), 
                                'r-', linewidth=1.5, alpha=0.8, label=self.proxy2_name)
        
        ax1.set_xlabel('Age (kyr ago)')
        ax1.set_ylabel(f'{self.proxy1_name}', color='blue')
        ax1_twin.set_ylabel(f'{self.proxy2_name}', color='red')
        ax1.set_title('Time Series')
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')
        
        # 2. Rolling correlation with interest areas
        ax2 = axes[0, 1]
        ax2.plot(self.rolling_correlation['age_kyr'], 
                self.rolling_correlation['rolling_correlation'],
                'purple', linewidth=2, label='Rolling Correlation')
        
        # Reference lines
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='High Corr. (+)')
        ax2.axhline(y=-0.7, color='orange', linestyle='--', alpha=0.7, label='High Corr. (-)')
        ax2.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Low threshold')
        ax2.axhline(y=-0.2, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Age (kyr)')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Rolling Window Correlation')
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.invert_xaxis()
        
        # 3. Histogram of correlation
        ax3 = axes[1, 0]
        corr_values = self.rolling_correlation['rolling_correlation']
        n, bins, patches = ax3.hist(corr_values, bins=30, alpha=0.7, 
                                    color='skyblue', edgecolor='black')
        
        # Color bars by category
        for patch, bin_val in zip(patches, bins[:-1]):
            if bin_val > 0.7:
                patch.set_color('green')
            elif bin_val < -0.7:
                patch.set_color('orange')
            elif abs(bin_val) < 0.2:
                patch.set_color('red')
        
        ax3.axvline(corr_values.mean(), color='red', linestyle='--', 
                    linewidth=2, label='Mean')
        
        # Adding elements for the legend with explanation of the colors
        legend_elements = [
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean'),
            Patch(facecolor='red', alpha=0.7, label='Decoupling'),
            Patch(facecolor='skyblue', alpha=0.7, label='Weak/moderate correlation'),
            Patch(facecolor='green', alpha=0.7, label='High positive correlation'),
            Patch(facecolor='orange', alpha=0.7, label='High negative correlation')
        ]
        
        ax3.set_xlabel('Correlation')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Correlation')
        ax3.grid(True, alpha=0.3)
        ax3.legend(handles=legend_elements, fontsize=8)
        
        # 4. Scatter plot 1st vs 2nd proxy colored by correlation
        ax4 = axes[1, 1]
        scatter = ax4.scatter(self.rolling_correlation['proxy1_values'], 
                                self.rolling_correlation['proxy2_values'].round(3),
                                c=self.rolling_correlation['rolling_correlation'],
                                cmap='RdBu_r', s=20, alpha=0.6)
        
        # General regression line
        slope, intercept, r_value, p_value, _ = stats.linregress(
            self.rolling_correlation['proxy1_values'], 
            self.rolling_correlation['proxy2_values']
        )
        
        x_range = np.array([self.rolling_correlation['proxy1_values'].min(), 
                            self.rolling_correlation['proxy1_values'].max()])
        y_line = slope * x_range + intercept
        ax4.plot(x_range, y_line, 'k--', alpha=0.8, 
                label=f'r = {r_value:.3f}, p = {p_value:.3e}')
        
        ax4.set_xlabel(f'{self.proxy1_name}')
        ax4.set_ylabel(f'{self.proxy2_name}')
        ax4.set_title(f'Relationship {self.proxy1_name} vs {self.proxy2_name}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Rolling Correlation')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/analysis_window_{window_size}kyrs.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure saved: {filename}")
        
    def plot_temporal_evolution(self, window_size: int = config.ROLLING_WINDOW_ANALYSIS['window_size'], figsize: Tuple[int, int] = config.ROLLING_WINDOW_PLOTS['temporal_evolution_figsize']) -> None:
        """
        Plot of temporal evolution with areas colored by period and save to file
        
        Parameters:
        -----------
        window_size : int
            Size of the window for the title
            Default: 41 kyr (Milankovitch obliquity cycle)
        figsize : tuple
            Size of the figure
        """
        if self.rolling_correlation is None:
            raise ValueError("Execute calculate_rolling_correlation() first!")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Time series in the upper panel
        ax1_twin = ax1.twinx()
        ax1.plot(self.interpolated_data['age_kyr'], 
                self.interpolated_data['proxy1_values'], 
                'b-', linewidth=2, alpha=0.8, label=self.proxy1_name)
        ax1_twin.plot(self.interpolated_data['age_kyr'], 
                        self.interpolated_data['proxy2_values'].round(3), 
                        'r-', linewidth=2, alpha=0.8, label=self.proxy2_name)
        
        ax1.set_ylabel(f'{self.proxy1_name}', color='blue')
        ax1_twin.set_ylabel(f'{self.proxy2_name}', color='red')
        ax1.set_title('Temporal Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Rolling correlation with filled areas in the lower panel
        corr_data = self.rolling_correlation
        ages = corr_data['age_kyr']
        corr = corr_data['rolling_correlation']
        
        ax2.plot(ages, corr, 'purple', linewidth=3, label=f'Rolling Correlation')
        
        # Filling areas by category
        ax2.fill_between(ages, 0, corr, where=(corr > 0.7), 
                        color='green', alpha=0.3, label='High Positive Correlation')
        ax2.fill_between(ages, 0, corr, where=(corr < -0.7), 
                        color='orange', alpha=0.3, label='High Negative Correlation')
        ax2.fill_between(ages, -0.2, 0.2, where=(abs(corr) < 0.2), 
                        color='red', alpha=0.2, label='Decoupling')
        
        # Reference lines
        for y_val, color, style in [(0, 'black', '-'), (0.7, 'green', '--'), 
                                    (-0.7, 'orange', '--'), (0.2, 'gray', ':'), 
                                    (-0.2, 'gray', ':')]:
            ax2.axhline(y=y_val, color=color, linestyle=style, alpha=0.5)
        
        ax2.set_xlabel('Age (kyr ago)')
        ax2.set_ylabel('Correlation')
        ax2.set_title(f'Rolling Correlation (Window: {window_size} kyr)')
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.invert_xaxis()
        
        # Set x-axis ticks using configured interval
        tick_interval = config.ROLLING_WINDOW_PLOTS['temporal_evolution_x_tick_interval']
        min_age = min(ages.min(), self.interpolated_data['age_kyr'].min())
        max_age = max(ages.max(), self.interpolated_data['age_kyr'].max())
        # Round to nearest tick_interval for clean ticks
        tick_start = int(min_age // tick_interval) * tick_interval
        tick_end = int(max_age // tick_interval + 1) * tick_interval
        x_ticks = np.arange(tick_start, tick_end + tick_interval, tick_interval)
        ax1.set_xticks(x_ticks)
        ax2.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks, rotation=90)
        ax2.set_xticklabels(x_ticks, rotation=90)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/temporal_evolution_window_{window_size}kyrs.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure saved: {filename}")
        
    def compare_window_sizes(self, window_sizes: List[int] = config.ROLLING_WINDOW_ANALYSIS['window_comparison_sizes'], figsize: Tuple[int, int] = config.ROLLING_WINDOW_PLOTS['window_comparison_figsize']) -> None:
        """
        Compare rolling correlation for 4 different window sizes
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes to compare
            Default: [20, 41, 100, 400] kyr (Milankovitch cycles)
            - 20 kyr: Precession (~19-23 kyr)
            - 41 kyr: Obliquity (~41 kyr) 
            - 100 kyr: Short eccentricity (~100 kyr)
            - 400 kyr: Long eccentricity (~400 kyr)
        figsize : tuple
            Size of the figure
        """
        if self.interpolated_data is None:
            raise ValueError("Execute interpolate_to_common_grid() first!")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Names of cycles for the titles (from configuration)
        cycle_names = config.ROLLING_WINDOW_ANALYSIS['cycle_names']
        
        for i, window in enumerate(window_sizes[:4]):  # Maximum 4 windows
            print(f"🔄 Calculating rolling correlation for window of {window} kyr")
            # Calculating correlation for this window
            rolling_corr = self.interpolated_data['proxy1_values'].rolling(
                window=window, min_periods=max(10, window//2)
            ).corr(self.interpolated_data['proxy2_values'])
            
            valid_data = pd.DataFrame({
                'age': self.interpolated_data['age_kyr'],
                'corr': rolling_corr
            }).dropna()
            
            ages = valid_data['age']
            corr = valid_data['corr']
            
            # Rolling correlation in purple
            axes[i].plot(ages, corr, 'purple', linewidth=3, label='Rolling Correlation')
            
            # Filling areas by category (same as temporal_evolution)
            axes[i].fill_between(ages, 0, corr, where=(corr > 0.7), 
                                color='green', alpha=0.3, label='High Positive Correlation')
            axes[i].fill_between(ages, 0, corr, where=(corr < -0.7), 
                                color='orange', alpha=0.3, label='High Negative Correlation')
            axes[i].fill_between(ages, -0.2, 0.2, where=(abs(corr) < 0.2), 
                                color='red', alpha=0.2, label='Decoupling')
            
            # Reference lines
            for y_val, color, style in [(0, 'black', '-'), (0.7, 'green', '--'), 
                                        (-0.7, 'orange', '--'), (0.2, 'gray', ':'), 
                                        (-0.2, 'gray', ':')]:
                axes[i].axhline(y=y_val, color=color, linestyle=style, alpha=0.5)
            
            axes[i].set_xlabel('Age (kyr ago)')
            axes[i].set_ylabel('Correlation')
            axes[i].set_title(f'{cycle_names[i]}')
            axes[i].set_ylim(-1, 1)
            axes[i].grid(True, alpha=0.3)
            axes[i].invert_xaxis()
            
            # Statistics
            mean_corr = corr.mean()
            std_corr = corr.std()
            axes[i].text(0.02, 0.98, f'μ = {mean_corr:.3f}\nσ = {std_corr:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend only in the last subplot to not pollute
            if i == 3:
                axes[i].legend(loc='lower right', fontsize=8)
        
        plt.suptitle(f'Comparison of Windows - {self.proxy1_name} vs {self.proxy2_name}', 
                        fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/windows_comparison.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure saved: {filename}")