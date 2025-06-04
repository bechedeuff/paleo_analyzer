# Paleoclimate Analysis Suite

A tool for analyzing paleoclimate proxy relationships using both rolling window correlation and spectral analysis methods. Designed to identify periods of coupling/decoupling and orbital-scale cyclicity in paleoclimate data.

## Features

### Core Capabilities
- **Dual Analysis**: Combined rolling window correlation + wavelet spectral analysis
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

### Visualizations
- **Rolling Window Plots**: Time series, correlation evolution, distribution analysis
- **Spectral Plots**: Wavelet power spectra, coherence analysis, global spectra
- **Cross-Analysis**: Phase relationships and cycle identification
- **Professional Output**: Publication-ready figures with proper scaling and annotations

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

### Plot Configurations
- `COMPREHENSIVE_ANALYSIS['figsize']`: Size for rolling window 4-subplot analysis
- `TEMPORAL_EVOLUTION['x_tick_interval']`: X-axis tick spacing in kyr
- `WAVELET_POWER_PLOT['figsize']`: Size for wavelet power plots
- `CROSS_WAVELET_PLOT['figsize']`: Size for cross-wavelet analysis
- `GLOBAL_SPECTRUM_PLOT['figsize']`: Size for global spectrum plots

## Output Files

Each execution creates a new experiment folder in `results/` (e.g., `experiment_1`, `experiment_2`, etc.) with organized subdirectories for both analyses.

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
│   │       ├── comprehensive_analysis_window_41kyrs.png
│   │       ├── temporal_evolution_window_41kyrs.png
│   │       └── windows_comparison.png
│   └── spectral/                              # Spectral analysis results
│       ├── spectral_analysis_report.pdf
│       ├── spectral_analysis_metadata.json
│       ├── spectral_analysis_results.csv
│       └── figures/
│           ├── comprehensive_wavelet_analysis_proxy1.png
│           ├── comprehensive_wavelet_analysis_proxy2.png
│           ├── cross_wavelet_analysis.png
│           └── global_wavelet_spectra.png
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
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- reportlab

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

### Integration of Both Methods
- **Rolling Window**: Identifies coupling/decoupling in the time domain
- **Spectral Analysis**: Identifies periodicities and frequency-domain relationships
- **Combined Insight**: Time-scale dependent relationships and orbital forcing mechanisms


## Scientific Applications

### Paleoclimatology
- Proxy relationship analysis (δ¹³C, δ¹⁸O, trace elements)
- Orbital forcing identification
- Climate system coupling/decoupling studies
- Time-scale dependent relationships

### Earth System Science
- Multi-proxy correlation analysis
- Frequency-domain relationship studies
- Lead-lag relationship identification
- Climate sensitivity to orbital forcing

## Citation

If you use this tool in your research, please cite appropriately and mention the key methods:
- Rolling window correlation analysis
- Continuous wavelet transform analysis using PyCWT
- Cross-wavelet coherence analysis
- Milankovitch cycle identification

**Citation:**
```
Bernardo S. Chede. Paleoclimate Analysis Suite. (2025). V. 0.0.1-beta. https://github.com/bechedeuff/paleo-analysis-suite
```

This tool uses the PyCWT (Python Continuous Wavelet Transform) library for spectral analysis. If you publish results using the spectral analysis features, please cite:

**Citation PyCWT:**
```
Sebastian Krieger and Nabil Freij. PyCWT: wavelet spectral analysis in Python. V. 0.4.0-beta. Python. 2023. https://github.com/regeirk/pycwt.
```

### 

## License

This project is open source. Feel free to modify and distribute according to your needs.
