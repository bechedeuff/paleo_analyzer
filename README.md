# Paleoclimate Analysis Suite

A tool for analyzing paleoclimate proxy relationships using rolling window correlation, spectral analysis, and lead-lag analysis methods. Designed to identify periods of coupling/decoupling, orbital-scale cyclicity, and temporal relationships in paleoclimate data.

## Features

### Core Capabilities
- **Triple Analysis**: Combined rolling window correlation + wavelet spectral analysis + lead-lag analysis
- **Generic Proxy Support**: Works with any paleoclimate proxy data (δ¹³C, ln(Mn), temperature, precipitation, etc.)
- **Configuration-Based**: All parameters controlled via configuration file - no interactive prompts
- **Automatic Column Detection**: Automatically detects proxy names from CSV column headers
- **Organized Output**: Structured experiment directories with separate analysis results

### Rolling Window Analysis
- **Configurable Windows**: Window sizes aligned with Milankovitch cycles (20, 41, 100, 400 kyr)
- **Period Classification**: Automatic identification of coupling, decoupling, and anti-correlation periods
- **Multi-Window Comparison**: Simultaneous analysis across different temporal scales

### Spectral Analysis
- **Continuous Wavelet Transform**: Advanced spectral analysis using PyCWT library
- **Milankovitch Cycle Detection**: Automatic identification of orbital cycles (precession, obliquity, eccentricity)
- **Coherence Analysis**: Cross-wavelet coherence and phase relationships
- **Statistical Significance**: Red noise testing with 95% confidence intervals

### Lead-Lag Analysis
- **Multiple Methods**: Standard cross-correlation, CCF AUC (Area Under Curve), and CCF at maximum lag
- **Robust Correlation Types**: Pearson, Spearman, and Kendall correlations
- **Bootstrap Confidence Intervals**: Statistical significance testing using scipy.stats.bootstrap
- **Data Preprocessing**: Optional detrending and normalization with automatic preprocessing pipeline
- **Temporal Relationship Detection**: Identifies which proxy leads/lags and by how much time
- **Comprehensive Visualization**: Multi-panel plots with time series, contrast analysis, and cross-correlation functions

### Visualizations
- **Rolling Window Plots**: Time series, correlation evolution, distribution analysis, multi-window comparisons
- **Spectral Plots**: Wavelet power spectra, coherence analysis, global spectra, cross-wavelet analysis
- **Lead-Lag Plots**: Time series with leader/lagger identification, contrast analysis, cross-correlation functions
- **Cross-Analysis**: Phase relationships, cycle identification, and temporal lead-lag relationships
- **Output**: Figures with proper scaling, annotations, and statistical significance indicators

## Quick Start

1. **Prepare your data**: Place CSV files in the `data/` folder
   - The CSV should be separated by commas
   - Format: First column = age (kyr), Second column = proxy values
   - Example: `data/d13C_age_cibicides.csv`, `data/lnmn-sue-TD-age.csv`

2. **Configure analysis**: Edit `src/configurations.py`
   ```python
   PROXY1_FILE = "file_1.csv"
   PROXY2_FILE = "file_2.csv"
   
   WINDOW_SIZE = 41  # kyr
   INTERPOLATION_RESOLUTION = 1.0  # kyr
   THRESHOLD_HIGH = 0.7
   THRESHOLD_LOW = 0.2
   ```

3. **Run analysis**:
   ```bash
   python main.py
   ```

The tool will automatically run all three analyses (rolling window, spectral, and lead-lag) and generate comprehensive results in organized experiment directories.

## Analysis Workflow

The tool follows this systematic workflow:

1. **Data Loading & Validation**: CSV files are loaded with automatic column detection
2. **Data Preprocessing**: Interpolation to common temporal grid, optional detrending/normalization
3. **Rolling Window Analysis**: Time-domain correlation analysis with multiple window sizes
4. **Spectral Analysis**: Frequency-domain analysis with wavelet transforms and coherence
5. **Lead-Lag Analysis**: Temporal precedence analysis with multiple methods and statistical testing
6. **Results Export**: PDF reports, JSON metadata, CSV data, and figures

Each analysis method provides complementary insights:
- **Rolling Window**: *When* are the proxies coupled/decoupled?
- **Spectral Analysis**: *What frequencies* show coherent behavior?
- **Lead-Lag Analysis**: *Which proxy leads* and *by how much time*?

## Configuration Options

All analysis parameters are controlled via `src/configurations.py`:

### Data Configuration
- `PROXY1_FILE`: First CSV file name (in data/ folder)
- `PROXY2_FILE`: Second CSV file name (in data/ folder)

### General Settings
- `INTERPOLATION_RESOLUTION`: Temporal resolution in kyr (default: 1.0)
- `MATPLOTLIB_STYLE`: Plot style (default: 'seaborn-v0_8')
- `SEABORN_PALETTE`: Color palette (default: "husl")
- `DEFAULT_FIGURE_SIZE`: Default figure dimensions
- `DEFAULT_FONT_SIZE`: Default font size
- `DEFAULT_DPI`: Figure resolution (default: 300)

### Rolling Window Analysis
- `WINDOW_SIZE`: Rolling window size in kyr (default: 41)
- `MIN_PERIODS`: Minimum periods for correlation calculation (default: None = auto)
- `THRESHOLD_HIGH`: High correlation threshold (default: 0.7)
- `THRESHOLD_LOW`: Decoupling threshold (default: 0.2)
- `WINDOW_COMPARISON['window_sizes']`: Window sizes for comparison [20, 41, 100, 400]
- `WINDOW_COMPARISON['cycle_names']`: Custom names for comparison plots

### Spectral Analysis
- `WAVELET_TYPE`: Type of wavelet ('morlet', 'paul', 'dog') (default: 'morlet')
- `WAVELET_PARAM`: Wavelet parameter (default: 6 for Morlet)
- `MIN_PERIOD`: Minimum period to analyze in kyr (default: 2.0)
- `MAX_PERIOD`: Maximum period to analyze in kyr (default: 1000.0)
- `COHERENCE_THRESHOLD`: Coherence threshold for significance (default: 0.7)
- `CONFIDENCE_LEVEL`: Statistical confidence level (default: 0.95)
- `MILANKOVITCH_CYCLES`: Dictionary defining orbital cycle periods

### Lead-Lag Analysis
- `LEADLAG_ANALYSIS['max_lag_kyr']`: Maximum lag to test in both directions (default: 50)
- `LEADLAG_ANALYSIS['lag_step_kyr']`: Step size for lag testing (default: 1.0)
- `LEADLAG_ANALYSIS['methods']`: List of methods to compute:
  - `'cross_correlation'`: Standard cross-correlation analysis
  - `'ccf_auc'`: Cross-correlation AUC (Area Under Curve) method
  - `'ccf_at_max_lag'`: Cross-correlation at maximum lag method
- `LEADLAG_ANALYSIS['correlation_types']`: Types of correlation (default: ['pearson', 'spearman', 'kendall'])
- `LEADLAG_ANALYSIS['significance_level']`: Statistical significance level (default: 0.05)
- `LEADLAG_ANALYSIS['confidence_level']`: Confidence level for bootstrap intervals (default: 0.95)
- `LEADLAG_ANALYSIS['bootstrap_iterations']`: Number of bootstrap iterations (default: 1000)
- `LEADLAG_ANALYSIS['detrend_data']`: Whether to detrend data before analysis (default: True)
- `LEADLAG_ANALYSIS['normalize_data']`: Whether to normalize data (z-score) (default: True)

### Plot Configurations
- `COMPREHENSIVE_ANALYSIS['figsize']`: Size for rolling window 4-subplot analysis
- `TEMPORAL_EVOLUTION['x_tick_interval']`: X-axis tick spacing in kyr
- `WAVELET_POWER_PLOT['figsize']`: Size for wavelet power plots
- `CROSS_WAVELET_PLOT['figsize']`: Size for cross-wavelet analysis
- `GLOBAL_SPECTRUM_PLOT['figsize']`: Size for global spectrum plots
- `LEADLAG_PLOTS['comprehensive_figsize']`: Size for comprehensive lead-lag analysis (default: (16, 10))
- `LEADLAG_PLOTS['default_correlation_type']`: Default correlation type for plots (default: 'pearson')
- `LEADLAG_PLOTS['contrast_line_color']`: Color for contrast line in plots (default: 'darkred')

## Output Files

Each execution creates a new experiment folder in `results/` (e.g., `experiment_1`, `experiment_2`, etc.) with organized subdirectories for all three analyses.

### Structure
```
results/
├── experiment_1/
│   ├── experiment_config.json                 # Configuration used (reproducibility)
│   ├── rolling_window/                        # Rolling window analysis results
│   │   ├── correlation_analysis_report.pdf
│   │   ├── analysis_metadata.json
│   │   ├── rolling_correlation_results.csv
│   │   └── figures/
│   │       ├── analysis_window_{window_size}kyrs.png
│   │       ├── temporal_evolution_window_{window_size}kyrs.png
│   │       └── windows_comparison.png
│   ├── spectral/                              # Spectral analysis results
│   │   ├── spectral_analysis_report.pdf
│   │   ├── spectral_analysis_metadata.json
│   │   ├── spectral_analysis_results.csv
│   │   └── figures/
│   │       ├── wavelet_analysis_proxy1.png
│   │       ├── wavelet_analysis_proxy2.png
│   │       ├── cross_wavelet_analysis.png
│   │       └── global_wavelet_spectra.png
│   └── lead_lag/                              # Lead-lag analysis results
│       ├── leadlag_analysis_report.pdf
│       ├── leadlag_analysis_metadata.json
│       ├── leadlag_analysis_results.csv
│       └── figures/
│           └── comprehensive_leadlag_analysis.png
├── experiment_2/
│   └── ...
```

### Rolling Window Files
- `correlation_analysis_report.pdf`: PDF report with correlation statistics and interpretation
- `analysis_metadata.json`: Metadata and detailed correlation results
- `rolling_correlation_results.csv`: Raw correlation data for further analysis
- `figures/`: Rolling window visualizations (time series, correlation evolution, comparisons)

### Spectral Analysis Files
- `spectral_analysis_report.pdf`: PDF report with identified Milankovitch cycles
- `spectral_analysis_metadata.json`: Spectral statistics and cycle identification results
- `spectral_analysis_results.csv`: Frequency-domain data (periods, power, coherence)
- `figures/`: Wavelet visualizations (power spectra, coherence, cross-wavelet analysis)

### Lead-Lag Analysis Files
- `leadlag_analysis_report.pdf`: PDF report with lead-lag analysis results and method comparisons
- `leadlag_analysis_metadata.json`: Metadata with analysis parameters and summary results
- `leadlag_analysis_results.csv`: Cross-correlation function results across all lags and correlation types
- `figures/comprehensive_leadlag_analysis.png`: Multi-panel visualization with time series, contrast, and cross-correlation

### Common Files
- `experiment_config.json`: Complete configuration used for this experiment (for reproducibility)

## Example Usage

### Basic Usage
```python
# Edit src/configurations.py with your settings
python main.py
```

## Dependencies

### Core Libraries
- pandas (data handling and CSV processing)
- numpy (numerical computations)
- matplotlib (plotting and visualization)
- seaborn (statistical data visualization)
- scipy (statistical analysis and bootstrap confidence intervals)
- reportlab (PDF report generation)

### Spectral Analysis
- pycwt (for continuous wavelet transforms)

## How to use the tool

1. Clone the repository
2. Create a virtual environment (or use conda)
```bash
python -m venv venv (venv)
conda create -n paleo_analysis python=3.12.7 (conda)
```

3. Activate the virtual environment
```bash
source venv/bin/activate (Linux)
.venv\Scripts\activate (Windows)
conda activate paleo_analysis (conda)
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Data Format

CSV files should follow this format (separated by commas):
```
age_kyr,proxy_name
0.1,23.45
0.2,23.52
0.3,23.41
...
```

- **First column**: Age in thousands of years (kyr)
- **Second column**: Proxy values (any units)
- **Header**: Column names are automatically detected

## Configuration Validation

The system automatically validates configurations and shows warnings for:
- Missing data files
- Invalid parameter ranges (correlation thresholds, window sizes, period ranges)
- Mismatched array lengths
- Potentially problematic settings
- Spectral analysis parameter consistency

## Customization

### Adding New Analysis Types
Extend the existing analyzer classes or create new ones following the established patterns.

### Custom Wavelets
Modify `WAVELET_TYPE` and `WAVELET_PARAM` in configurations for different mother wavelets.

### Different Time Scales
Adjust `INTERPOLATION_RESOLUTION`, `MIN_PERIOD`, and `MAX_PERIOD` for your specific time scales.

## Understanding the Results

### Rolling Window Analysis Results
1. **Comprehensive Analysis Plot**: Overview with time series, correlation evolution, distribution, and scatter plot
2. **Temporal Evolution Plot**: Detailed view of correlation changes through time with colored periods
3. **Window Comparison Plot**: How correlation patterns change with different window sizes

### Spectral Analysis Results
1. **Comprehensive Wavelet Analysis (per proxy)**: Individual wavelet power spectra with reconstruction and global spectrum
2. **Cross-Wavelet Analysis**: Combined analysis showing cross-power, coherence, phase relationships, and global coherence
3. **Global Wavelet Spectra**: Comparison of power spectra and coherence across all frequencies

### Lead-Lag Analysis Results
1. **Lead-Lag Plot**: Multi-panel visualization showing:
2. **Quantitative Results**: 
   - **Cross-correlation method**: Optimal lag and correlation strength
   - **CCF AUC method**: Area under curve measure for lead-lag assessment
   - **CCF at maximum lag**: Correlation at the lag with maximum absolute correlation
3. **Statistical Confidence**: Bootstrap confidence intervals for all methods
4. **Multiple Correlation Types**: Results for Pearson, Spearman, and Kendall correlations

## Scientific Applications

### Paleoclimatology
- Proxy relationship analysis (δ¹³C, δ¹⁸O, trace elements)
- Orbital forcing identification
- Climate system coupling/decoupling studies
- Time-scale dependent relationships

### Earth System Science
- Multi-proxy correlation analysis
- Frequency-domain relationship studies
- Temporal precedence and causality studies
- Lead-lag relationship identification
- Climate sensitivity to orbital forcing
- Process timing and synchronization analysis

## Citation

If you use this tool in your research, please cite appropriately:

**Citation:**
```
Bernardo S. Chede. Paleoclimate Analysis Suite. (2025). V. 0.0.1-beta. https://github.com/bechedeuff/paleo-analysis-suite
```

This tool uses the PyCWT (Python Continuous Wavelet Transform) library for spectral analysis. If you publish results using the spectral analysis features, please cite:

**Citation PyCWT:**
```
Sebastian Krieger and Nabil Freij. PyCWT: wavelet spectral analysis in Python. V. 0.4.0-beta. Python. 2023. https://github.com/regeirk/pycwt.
```

**Citation Lead-Lag**

The lead-lag analysis implementation was inspired: https://github.com/philipperemy/lead-lag

### 

## License

This project is open source. Feel free to modify and utilize according to your needs.
