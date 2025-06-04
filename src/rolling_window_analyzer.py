from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
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

class PaleoclimateCorrelationAnalyzer:
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
        
        # Linear interpolation for 1st proxy
        interp_proxy1 = interp1d(
            self.proxy1_data['age_kyr'], 
            self.proxy1_data['proxy1_values'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Linear interpolation for the 2nd proxy
        interp_proxy2 = interp1d(
            self.proxy2_data['age_kyr'],
            self.proxy2_data['proxy2_values'],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Creating DF with interpolated data
        self.interpolated_data = pd.DataFrame({
            'age_kyr': common_ages,
            'proxy1_values': interp_proxy1(common_ages),
            'proxy2_values': interp_proxy2(common_ages)
        }).dropna().reset_index(drop=True)
        
        print(f"✅ Interpolation completed: {len(self.interpolated_data)} points")
        print(f"   Final range: {self.interpolated_data['age_kyr'].min():.1f} - {self.interpolated_data['age_kyr'].max():.1f} kyr")
        
    def calculate_rolling_correlation(self, window_size: int = config.WINDOW_SIZE, min_periods: Optional[int] = config.MIN_PERIODS) -> None:
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
        
    def identify_correlation_periods(self, threshold_high: float = config.THRESHOLD_HIGH, threshold_low: float = config.THRESHOLD_LOW) -> Dict[str, pd.DataFrame]:
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
        
    def plot_comprehensive_analysis(self, window_size: int = config.WINDOW_SIZE, figsize: Tuple[int, int] = config.COMPREHENSIVE_ANALYSIS['figsize']) -> None:
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
        ax1.legend(lines, labels, loc='upper left')
        
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
        filename = f'{self.experiment_dir}/figures/comprehensive_analysis_window_{window_size}kyrs.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Figure saved: {filename}")
        
    def plot_temporal_evolution(self, window_size: int = config.WINDOW_SIZE, figsize: Tuple[int, int] = config.TEMPORAL_EVOLUTION['figsize']) -> None:
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
        tick_interval = config.TEMPORAL_EVOLUTION['x_tick_interval']
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
        
    def compare_window_sizes(self, window_sizes: List[int] = config.WINDOW_COMPARISON['window_sizes'], figsize: Tuple[int, int] = config.WINDOW_COMPARISON['figsize']) -> None:
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
        cycle_names = config.WINDOW_COMPARISON['cycle_names']
        
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
        
    def export_results(self, filename: str = 'rolling_correlation_results.csv') -> None:
        """
        Export results to CSV file
        
        Parameters:
        -----------
        filename : str
            Name of the file to export
        """
        if self.rolling_correlation is None:
            raise ValueError("Execute calculate_rolling_correlation() first!")
            
        # Create full path with experiment directory
        full_path = f'{self.experiment_dir}/{filename}'
            
        # Create export DataFrame with original column names
        export_data = self.rolling_correlation.copy()
        export_data = export_data.rename(columns={
            'proxy1_values': self.proxy1_name,
            'proxy2_values': self.proxy2_name
        })
        
        export_data.to_csv(full_path, index=False, sep=';', decimal=',')
        print(f"✅ Results exported to: {full_path}")

    def create_pdf_report(self, window_size: int, periods: Dict[str, pd.DataFrame]) -> str:
        """
        Create a well-formatted PDF report
        
        Parameters:
        -----------
        window_size : int
            Window size used in the analysis
        periods : dict
            Dictionary with classified periods
            
        Returns:
        --------
        str
            Path to the generated PDF file
        """
        # Prepare data
        corr_stats = self.rolling_correlation['rolling_correlation']
        analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # PDF configuration
        pdf_file = f'{self.experiment_dir}/correlation_analysis_report.pdf'
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom style for title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.black
        )
        
        # Section style
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.black
        )
        
        # Main title
        story.append(Paragraph("REPORT OF ROLLING WINDOW CORRELATION ANALYSIS", title_style))
        story.append(Spacer(1, 20))
        
        # General information
        info_data = [
            ['Analysis date:', analysis_date],
            ['Proxy 1:', self.proxy1_name],
            ['Proxy 2:', self.proxy2_name],
            ['Window used:', f'{window_size} kyr'],
            ['Analyzed period:', f"{self.interpolated_data['age_kyr'].min():.1f} - {self.interpolated_data['age_kyr'].max():.1f} kyr ago"],
            ['Total points:', str(len(self.rolling_correlation))]
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
        
        # Statistics
        story.append(Paragraph("STATISTICS OF CORRELATION", section_style))
        stats_data = [
            ['Mean correlation:', f'{corr_stats.mean():.3f}'],
            ['Median correlation:', f'{corr_stats.median():.3f}'],
            ['Standard deviation:', f'{corr_stats.std():.3f}'],
            ['Minimum correlation:', f'{corr_stats.min():.3f}'],
            ['Maximum correlation:', f'{corr_stats.max():.3f}']
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Period count
        story.append(Paragraph("PERIOD COUNT", section_style))
        periods_data = [
            ['High positive correlation (r > 0.7):', f"{len(periods['high_positive'])} periods"],
            ['High negative correlation (r < -0.7):', f"{len(periods['high_negative'])} periods"],
            ['Decoupling (|r| < 0.2):', f"{len(periods['decoupled'])} periods"]
        ]
        
        periods_table = Table(periods_data, colWidths=[3*inch, 1.5*inch])
        periods_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen)
        ]))
        story.append(periods_table)
        story.append(Spacer(1, 20))
        
        # Main periods by category
        for category, name, color in [('high_positive', 'HIGH POSITIVE CORRELATION', colors.green),
                                        ('high_negative', 'HIGH NEGATIVE CORRELATION', colors.red),
                                        ('decoupled', 'DECOUPLING', colors.orange)]:
            if len(periods[category]) > 0:
                story.append(Paragraph(f"MAIN PERIODS - {name}", section_style))
                
                # Period data (maximum 10)
                period_data = [['Age (kyr ago)', 'Correlation']]
                for _, row in periods[category].head(10).iterrows():
                    period_data.append([f"{row['age_kyr']:.1f}", f"{row['rolling_correlation']:.3f}"])
                
                period_table = Table(period_data, colWidths=[1.5*inch, 1.5*inch])
                period_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, 0), color),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
                ]))
                story.append(period_table)
                story.append(Spacer(1, 15))
        
        # Interpretation
        story.append(Paragraph("POSSIBLE INTERPRETATIONS", section_style))
        interpretation_text = f"""
        <b>• POSITIVE CORRELATION:</b> {self.proxy1_name} and {self.proxy2_name} vary together
        (synchronized processes)<br/><br/>
        
        <b>• NEGATIVE CORRELATION:</b> Variables move in opposite directions
        (antagonistic or out of phase processes)<br/><br/>
        
        <b>• DECOUPLING:</b> Independent evolution of variables
        (different dominant forcings)
        """
        
        story.append(Paragraph(interpretation_text, styles['Normal']))
        
        # Generate PDF
        doc.build(story)
        return pdf_file

    def create_json_metadata(self, window_size: int, periods: Dict[str, pd.DataFrame]) -> str:
        """
        Create a JSON file with structured metadata
        
        Parameters:
        -----------
        window_size : int
            Window size used in the analysis
        periods : dict
            Dictionary with classified periods
            
        Returns:
        --------
        str
            Path to the generated JSON file
        """
        corr_stats = self.rolling_correlation['rolling_correlation']
        
        # Structured data
        metadata = {
            "analysis_info": {
                "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "proxy1_name": self.proxy1_name,
                "proxy2_name": self.proxy2_name,
                "window_size_kyr": window_size,
                "analysis_period": {
                    "start_kyr": float(self.interpolated_data['age_kyr'].min()),
                    "end_kyr": float(self.interpolated_data['age_kyr'].max()),
                    "duration_kyr": float(self.interpolated_data['age_kyr'].max() - self.interpolated_data['age_kyr'].min())
                },
                "total_points": len(self.rolling_correlation)
            },
            "data_summary": {
                "proxy1": {
                    "name": self.proxy1_name,
                    "original_points": len(self.proxy1_data),
                    "age_range": {
                        "min_kyr": float(self.proxy1_data['age_kyr'].min()),
                        "max_kyr": float(self.proxy1_data['age_kyr'].max())
                    },
                    "value_range": {
                        "min": float(self.proxy1_data['proxy1_values'].min()),
                        "max": float(self.proxy1_data['proxy1_values'].max())
                    }
                },
                "proxy2": {
                    "name": self.proxy2_name,
                    "original_points": len(self.proxy2_data),
                    "age_range": {
                        "min_kyr": float(self.proxy2_data['age_kyr'].min()),
                        "max_kyr": float(self.proxy2_data['age_kyr'].max())
                    },
                    "value_range": {
                        "min": float(self.proxy2_data['proxy2_values'].min()),
                        "max": float(self.proxy2_data['proxy2_values'].max())
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
        json_file = f'{self.experiment_dir}/analysis_metadata.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return json_file


