import numpy as np

# Import modules
from src.rolling_window_analyzer import PaleoclimateRollingWindowAnalyzer
from src.spectral_analyzer import PaleoclimateSpectralAnalyzer
from src.lead_lag_analyzer import PaleoclimateLead_LagAnalyzer
from src import configurations as config
from src.utils import (
    create_results_directory, save_experiment_config, print_configuration_summary, validate_configurations,
    export_rolling_window_results, export_spectral_results, export_leadlag_results,
    load_paleoclimate_data, interpolate_to_common_grid
)
from src.metadata import (
    create_rolling_window_metadata, create_spectral_metadata, create_leadlag_metadata
)
from src.pdf_reports import (
    create_rolling_window_pdf_report, create_spectral_pdf_report, create_leadlag_pdf_report
)

def run_complete_analysis() -> None:
    """
    Run both rolling window and spectral analyses in sequence
    """
    print("🌍 PALEOCLIMATE ANALYSIS SUITE")
    print("=" * 70)
    print("Running both Rolling Window Correlation and Spectral Analysis")
    print("Results will be organized in experiment subdirectories")
    print("-" * 70)
    
    try:
        
        # Show configuration summary
        print_configuration_summary()
        
        # Validate configurations and show warnings
        warnings = validate_configurations()
        if warnings:
            print("⚠️  CONFIGURATION WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
            print()
        
        # Create experiment directory structure
        print("🔧 PREPARING ANALYSIS...")
        experiment_dir = create_results_directory()
        
        # Save current configuration for reproducibility
        save_experiment_config(experiment_dir)
        
        # Get file paths from configuration
        proxy1_file = f'data/{config.PROXY1_FILE}'
        proxy2_file = f'data/{config.PROXY2_FILE}'
        
        # =============================================================================
        # ROLLING WINDOW CORRELATION ANALYSIS
        # =============================================================================
        
        print("\n" + "="*70)
        print("🔄 ROLLING WINDOW CORRELATION ANALYSIS")
        print("="*70)
        
        # Initialize rolling window analyzer with specific subdirectory
        rolling_analyzer = PaleoclimateRollingWindowAnalyzer(f'{experiment_dir}/rolling_window')
        
        # Data loading
        print("\n📂 LOADING DATA FOR CORRELATION ANALYSIS...")
        success, data = load_paleoclimate_data(proxy1_file, proxy2_file)
        if not success or not data:
            print("❌ Error in data loading for rolling window analysis!")
            return
        
        # Set analyzer attributes
        rolling_analyzer.proxy1_data = data['proxy1_data']
        rolling_analyzer.proxy2_data = data['proxy2_data']
        rolling_analyzer.proxy1_name = data['proxy1_name']
        rolling_analyzer.proxy2_name = data['proxy2_name']
        rolling_analyzer.proxy1_units = data['proxy1_units']
        rolling_analyzer.proxy2_units = data['proxy2_units']
        
        # Data processing
        print("\n🔧 PROCESSING DATA...")
        rolling_analyzer.interpolated_data = interpolate_to_common_grid(
            rolling_analyzer.proxy1_data, rolling_analyzer.proxy2_data)
        
        # Calculate rolling correlation
        print(f"\n📊 CALCULATING ROLLING CORRELATION (window: {config.ROLLING_WINDOW_ANALYSIS['window_size']} kyr)...")
        rolling_analyzer.calculate_rolling_correlation()
        
        # Identify periods
        print("\n🔍 IDENTIFYING CORRELATION PERIODS...")
        periods = rolling_analyzer.identify_correlation_periods()
        
        # Generate visualizations
        print("\n📈 GENERATING ROLLING WINDOW VISUALIZATIONS...")
        
        print("   • Rolling window analysis...")
        rolling_analyzer.plot_comprehensive_analysis()
        
        print("   • Temporal evolution...")
        rolling_analyzer.plot_temporal_evolution()
        
        print("   • Window size comparison...")
        rolling_analyzer.compare_window_sizes()
        
        # Export rolling window results
        print("\n💾 EXPORTING ROLLING WINDOW RESULTS...")
        
        # Export CSV results
        if rolling_analyzer.rolling_correlation is not None:
            export_rolling_window_results(rolling_analyzer.rolling_correlation, rolling_analyzer.proxy1_name, 
                                            rolling_analyzer.proxy2_name, rolling_analyzer.experiment_dir, 
                                            'rolling_correlation_results.csv')
        
        # Create PDF report
        print("\n📄 Generating PDF report...")
        rolling_pdf = create_rolling_window_pdf_report(
            rolling_analyzer.rolling_correlation, rolling_analyzer.interpolated_data,
            rolling_analyzer.proxy1_name, rolling_analyzer.proxy2_name, 
            config.ROLLING_WINDOW_ANALYSIS['window_size'], periods, rolling_analyzer.experiment_dir
        )
        print(f"✅ Rolling window PDF report: {rolling_pdf}")
        
        # Create JSON metadata
        rolling_json = create_rolling_window_metadata(rolling_analyzer.rolling_correlation, rolling_analyzer.proxy1_data, 
                                                        rolling_analyzer.proxy2_data, rolling_analyzer.interpolated_data,
                                                        rolling_analyzer.proxy1_name, rolling_analyzer.proxy2_name,
                                                        config.ROLLING_WINDOW_ANALYSIS['window_size'], periods, 
                                                        rolling_analyzer.experiment_dir)
        print(f"✅ Rolling window metadata JSON: {rolling_json}")
        
        # Rolling window summary
        corr_stats = rolling_analyzer.rolling_correlation['rolling_correlation']
        print(f"\n📊 ROLLING WINDOW SUMMARY:")
        print(f"   Mean correlation: {corr_stats.mean():.3f} (range: {corr_stats.min():.3f} to {corr_stats.max():.3f})")
        print(f"   Periods found: {len(periods['high_positive'])} pos | {len(periods['high_negative'])} neg | {len(periods['decoupled'])} decoupled")
        
        # =============================================================================
        # SPECTRAL ANALYSIS
        # =============================================================================
        
        print("\n" + "="*70)
        print("🌊 SPECTRAL ANALYSIS")
        print("="*70)
        
        # Initialize spectral analyzer with specific subdirectory
        spectral_analyzer = PaleoclimateSpectralAnalyzer(f'{experiment_dir}/spectral')
        
        # Data loading (reuse same data)
        print("\n📂 LOADING DATA FOR SPECTRAL ANALYSIS...")
        print("🔄 Loading paleoclimate data for spectral analysis...")
        
        # Set analyzer attributes (reuse loaded data)
        spectral_analyzer.proxy1_data = data['proxy1_data']
        spectral_analyzer.proxy2_data = data['proxy2_data']
        spectral_analyzer.proxy1_name = data['proxy1_name']
        spectral_analyzer.proxy2_name = data['proxy2_name']
        spectral_analyzer.proxy1_units = data['proxy1_units']
        spectral_analyzer.proxy2_units = data['proxy2_units']
        
        # Data processing
        print("\n🔧 PROCESSING DATA FOR SPECTRAL ANALYSIS...")
        print("🔄 Interpolating to common grid for spectral analysis...")
        spectral_analyzer.interpolated_data = interpolate_to_common_grid(
            spectral_analyzer.proxy1_data, spectral_analyzer.proxy2_data)
        
        # Wavelet analysis
        print(f"\n🌊 WAVELET ANALYSIS ({config.SPECTRAL_ANALYSIS['wavelet_type']} wavelet)...")
        spectral_analyzer.calculate_wavelet_transform()
        
        # Identify Milankovitch cycles
        print("\n🔍 IDENTIFYING MILANKOVITCH CYCLES...")
        cycles_found = spectral_analyzer.identify_milankovitch_cycles()
        
        # Generate spectral visualizations
        print("\n📈 GENERATING SPECTRAL VISUALIZATIONS...")
        
        print("   • Wavelet analysis...")
        spectral_analyzer.plot_comprehensive_wavelet_analysis(proxy_num=1)
        spectral_analyzer.plot_comprehensive_wavelet_analysis(proxy_num=2)
        
        print("   • Cross-wavelet analysis...")
        spectral_analyzer.plot_cross_wavelet_analysis()
        
        print("   • Global wavelet spectra...")
        spectral_analyzer.plot_global_wavelet_spectra()
        
        # Export spectral results
        print("\n💾 EXPORTING SPECTRAL RESULTS...")
        
        # Export CSV results
        if spectral_analyzer.cwt1_power is not None:
            export_spectral_results(spectral_analyzer.cwt1_power, spectral_analyzer.cwt2_power, 
                                    spectral_analyzer.coherence, spectral_analyzer.periods, 
                                    spectral_analyzer.freqs, spectral_analyzer.proxy1_name, 
                                    spectral_analyzer.proxy2_name, spectral_analyzer.experiment_dir, 
                                    'spectral_analysis_results.csv')
        
        # Create PDF report
        print("\n📄 Generating spectral PDF report...")
        spectral_pdf = create_spectral_pdf_report(
            spectral_analyzer.interpolated_data, spectral_analyzer.periods,
            spectral_analyzer.proxy1_name, spectral_analyzer.proxy2_name,
            cycles_found, spectral_analyzer.experiment_dir
        )
        print(f"✅ Spectral analysis PDF report: {spectral_pdf}")
        
        # Create JSON metadata
        spectral_json = create_spectral_metadata(spectral_analyzer.interpolated_data, spectral_analyzer.cwt1_power, 
                                                spectral_analyzer.cwt2_power, spectral_analyzer.coherence, 
                                                spectral_analyzer.periods, spectral_analyzer.proxy1_name, 
                                                spectral_analyzer.proxy2_name, cycles_found, 
                                                spectral_analyzer.experiment_dir)
        print(f"✅ Spectral analysis metadata JSON: {spectral_json}")
        
        # Spectral summary
        total_cycles = sum(len(cycles) for cycles in cycles_found.values())
        mean_coherence = (np.mean(spectral_analyzer.coherence) 
                            if spectral_analyzer.coherence is not None and spectral_analyzer.coherence.size > 0 
                            else 0)
        print(f"\n🌊 SPECTRAL ANALYSIS SUMMARY:")
        print(f"   Frequency bands analyzed: {len(spectral_analyzer.periods)}")
        print(f"   Milankovitch cycles found: {total_cycles}")
        print(f"   Mean coherence: {mean_coherence:.3f}")
        
        # =============================================================================
        # LEAD-LAG ANALYSIS
        # =============================================================================
        
        print("\n" + "="*70)
        print("🔄 LEAD-LAG ANALYSIS")
        print("="*70)
        
        # Initialize lead-lag analyzer with specific subdirectory
        leadlag_analyzer = PaleoclimateLead_LagAnalyzer(f'{experiment_dir}/lead_lag')
        
        # Data loading (reuse same data)
        print("\n📂 LOADING DATA FOR LEAD-LAG ANALYSIS...")
        
        # Set analyzer attributes (reuse loaded data)
        leadlag_analyzer.proxy1_data = data['proxy1_data']
        leadlag_analyzer.proxy2_data = data['proxy2_data']
        leadlag_analyzer.proxy1_name = data['proxy1_name']
        leadlag_analyzer.proxy2_name = data['proxy2_name']
        leadlag_analyzer.proxy1_units = data['proxy1_units']
        leadlag_analyzer.proxy2_units = data['proxy2_units']
        
        # Data processing
        print("\n🔧 PROCESSING DATA FOR LEAD-LAG ANALYSIS...")
        leadlag_analyzer.interpolated_data = interpolate_to_common_grid(
            leadlag_analyzer.proxy1_data, leadlag_analyzer.proxy2_data)
        
        # Lead-lag analysis
        print(f"\n🔄 LEAD-LAG ANALYSIS (max lag: {config.LEADLAG_ANALYSIS['max_lag_kyr']} kyr)...")
        leadlag_results = leadlag_analyzer.calculate_comprehensive_leadlag_analysis()
        
        # Generate lead-lag visualizations
        print("\n📈 GENERATING LEAD-LAG VISUALIZATIONS...")
        
        print("   • Lead-lag analysis...")
        leadlag_analyzer.plot_comprehensive_leadlag_analysis()
        
        # Export lead-lag results
        print("\n💾 EXPORTING LEAD-LAG RESULTS...")
        
        # Export CSV results
        if leadlag_analyzer.leadlag_results is not None:
            export_leadlag_results(leadlag_analyzer.leadlag_results, leadlag_analyzer.experiment_dir, 
                                    'leadlag_analysis_results.csv')
        
        # Create PDF report
        print("\n📄 Generating lead-lag PDF report...")
        leadlag_pdf = create_leadlag_pdf_report(
            leadlag_analyzer.leadlag_results, leadlag_analyzer.interpolated_data,
            leadlag_analyzer.proxy1_name, leadlag_analyzer.proxy2_name,
            leadlag_analyzer.experiment_dir
        )
        print(f"✅ Lead-lag analysis PDF report: {leadlag_pdf}")
        
        # Create JSON metadata
        leadlag_json = create_leadlag_metadata(leadlag_analyzer.leadlag_results, leadlag_analyzer.interpolated_data,
                                                leadlag_analyzer.proxy1_name, leadlag_analyzer.proxy2_name, 
                                                leadlag_analyzer.experiment_dir)
        print(f"✅ Lead-lag analysis metadata JSON: {leadlag_json}")
        
        # Lead-lag summary
        methods_summary = []
        for method, method_data in leadlag_results['summary'].items():
            if 'pearson' in method_data:
                if method == 'cross_correlation':
                    lag = method_data['pearson']['optimal_lag_kyr']
                    corr = method_data['pearson']['optimal_correlation']
                    methods_summary.append(f"{method}: lag={lag:.1f} kyr, r={corr:.3f}")
                elif method == 'ccf_auc':
                    auc = method_data['pearson']['auc_measure']
                    methods_summary.append(f"{method}: AUC={auc:.3f}")
                    
        print(f"\n🔄 LEAD-LAG SUMMARY:")
        print(f"   Methods analyzed: {len(leadlag_results['analysis_info']['methods'])}")
        print(f"   Correlation types: {len(leadlag_results['analysis_info']['correlation_types'])}")
        for summary in methods_summary:
            print(f"   {summary}")
        
        
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure all required packages are installed:")
        print("   pip install pandas numpy matplotlib seaborn scipy reportlab pycwt")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    run_complete_analysis()