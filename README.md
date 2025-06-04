# Paleoclimate Rolling Window Correlation Analyzer

A tool for analyzing rolling window correlations between paleoclimate proxies, designed to identify periods of coupling and decoupling in paleoclimate data.

## Features

- **Generic Proxy Support**: Works with any type of paleoclimate proxy data (δ13C, ln(Mn), temperature, precipitation, etc.)
- **Configuration-Based**: All parameters controlled via configuration file - no interactive prompts
- **Automatic Column Detection**: Automatically detects proxy names from CSV column headers
- **Rolling Window Analysis**: Configurable window sizes with Milankovitch cycles as default
- **Output**: Generates PDF reports, JSON metadata, Experiment configuration and figures
- **Visualizations**: 
  - Time series plots with dual y-axes
  - Rolling correlation with color-coded periods
  - Correlation distribution histograms
  - Scatter plots with regression analysis
  - Multi-window comparison plots

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

### Analysis Parameters
- `WINDOW_SIZE`: Rolling window size in kyr (default: 41)
- `INTERPOLATION_RESOLUTION`: Temporal resolution in kyr (default: 1.0)
- `MIN_PERIODS`: Minimum periods for correlation calculation (default: None = auto)
- `THRESHOLD_HIGH`: High correlation threshold (default: 0.7)
- `THRESHOLD_LOW`: Decoupling threshold (default: 0.2)

### Visualization Settings
- `MATPLOTLIB_STYLE`: Plot style (default: 'seaborn-v0_8')
- `SEABORN_PALETTE`: Color palette (default: "husl")
- `DEFAULT_FIGURE_SIZE`: Default figure dimensions
- `DEFAULT_FONT_SIZE`: Default font size

### Plot-Specific Settings
- `COMPREHENSIVE_ANALYSIS['figsize']`: Size for 4-subplot analysis
- `TEMPORAL_EVOLUTION['x_tick_interval']`: X-axis tick spacing in kyr
- `WINDOW_COMPARISON['window_sizes']`: Window sizes for comparison
- `WINDOW_COMPARISON['cycle_names']`: Custom names for comparison plots

## Output Files

Each execution creates a new experiment folder in `results/` (e.g., `experiment_1`, `experiment_2`, etc.) to prevent overwriting previous results.

### Structure
```
results/
├── experiment_1/
│   ├── correlation_analysis_report.pdf
│   ├── analysis_metadata.json
│   ├── rolling_correlation_results.csv
│   ├── experiment_config.json
│   └── figures/
│       ├── comprehensive_analysis_window_Xkyrs.png
│       ├── temporal_evolution_window_Xkyrs.png
│       └── windows_comparison.png
├── experiment_2/
│   └── ...
```

### Files in Each Experiment
- `correlation_analysis_report.pdf`: PDF report with statistics and interpretation
- `analysis_metadata.json`: Machine-readable metadata and detailed results
- `rolling_correlation_results.csv`: Raw correlation data for further analysis
- `experiment_config.json`: Configuration used for this experiment (for reproducibility)
- `figures/`: All generated plots and visualizations

## Example Usage

### Basic Usage
```python
# Edit src/configurations.py with your settings
python main.py
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- reportlab

Install with:
```bash
pip install -r requirements.txt
```

## Data Format

CSV files should follow this format:
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

## Interpretation Guide

### Correlation Values
- **High Positive (r > 0.7)**: Proxies vary together (synchronized processes)
- **High Negative (r < -0.7)**: Proxies vary in opposition (antagonistic/out-of-phase)
- **Decoupling (|r| < 0.2)**: Independent evolution (different dominant forcings)

### Window Sizes
- **20 kyr**: Precession cycles (~19-23 kyr)
- **41 kyr**: Obliquity cycles (~41 kyr) - Default
- **100 kyr**: Short eccentricity cycles (~100 kyr)
- **400 kyr**: Long eccentricity cycles (~400 kyr)

## Configuration Validation

The system automatically validates configurations and shows warnings for:
- Missing data files
- Invalid parameter ranges
- Mismatched array lengths
- Potentially problematic settings

## Customization

### Adding New Plot Types
Extend the `PaleoclimateCorrelationAnalyzer` class with new visualization methods.

### Custom Correlation Metrics
Modify the `calculate_rolling_correlation` method to use different correlation measures.

### Different Time Scales
Adjust `INTERPOLATION_RESOLUTION` and `WINDOW_SIZE` for different temporal scales.

## License

This project is open source. Feel free to modify and distribute according to your needs.
