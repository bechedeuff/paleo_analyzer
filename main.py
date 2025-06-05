import os
from src.utils import create_results_directory, save_experiment_config, print_configuration_summary, validate_configurations
from src.rolling_window_analyzer import PaleoclimateCorrelationAnalyzer
from src.spectral_analyzer import PaleoclimateSpectralAnalyzer
from src.lead_lag_analyzer import PaleoclimateLead_LagAnalyzer
from src import configurations as config
import numpy as np

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
        rolling_analyzer = PaleoclimateCorrelationAnalyzer(f'{experiment_dir}/rolling_window')
        
        # Data loading
        print("\n📂 LOADING DATA FOR CORRELATION ANALYSIS...")
        if not rolling_analyzer.load_data(proxy1_file, proxy2_file):
            print("❌ Error in data loading for rolling window analysis!")
            return
        
        # Data processing
        print("\n🔧 PROCESSING DATA...")
        rolling_analyzer.interpolate_to_common_grid()
        
        # Calculate rolling correlation
        print(f"\n📊 CALCULATING ROLLING CORRELATION (window: {config.WINDOW_SIZE} kyr)...")
        rolling_analyzer.calculate_rolling_correlation()
        
        # Identify periods
        print("\n🔍 IDENTIFYING CORRELATION PERIODS...")
        periods = rolling_analyzer.identify_correlation_periods()
        
        # Generate visualizations
        print("\n📈 GENERATING ROLLING WINDOW VISUALIZATIONS...")
        
        print("   • Comprehensive analysis (4 subplots)...")
        rolling_analyzer.plot_comprehensive_analysis()
        
        print("   • Temporal evolution...")
        rolling_analyzer.plot_temporal_evolution()
        
        print("   • Window size comparison...")
        rolling_analyzer.compare_window_sizes()
        
        # Export rolling window results
        print("\n💾 EXPORTING ROLLING WINDOW RESULTS...")
        
        rolling_analyzer.export_results('rolling_correlation_results.csv')
        
        rolling_pdf = rolling_analyzer.create_pdf_report(config.WINDOW_SIZE, periods)
        print(f"✅ Rolling window PDF report: {rolling_pdf}")
        
        rolling_json = rolling_analyzer.create_json_metadata(config.WINDOW_SIZE, periods)
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
        
        # Data loading (reuse same files)
        print("\n📂 LOADING DATA FOR SPECTRAL ANALYSIS...")
        if not spectral_analyzer.load_data(proxy1_file, proxy2_file):
            print("❌ Error in data loading for spectral analysis!")
            return
        
        # Data processing
        print("\n🔧 PROCESSING DATA FOR SPECTRAL ANALYSIS...")
        spectral_analyzer.interpolate_to_common_grid()
        
        # Wavelet analysis
        print(f"\n🌊 WAVELET ANALYSIS ({config.WAVELET_TYPE} wavelet)...")
        spectral_analyzer.calculate_wavelet_transform()
        
        # Identify Milankovitch cycles
        print("\n🔍 IDENTIFYING MILANKOVITCH CYCLES...")
        cycles_found = spectral_analyzer.identify_milankovitch_cycles()
        
        # Generate spectral visualizations
        print("\n📈 GENERATING SPECTRAL VISUALIZATIONS...")
        
        print("   • Comprehensive wavelet analysis...")
        spectral_analyzer.plot_comprehensive_wavelet_analysis(proxy_num=1)
        spectral_analyzer.plot_comprehensive_wavelet_analysis(proxy_num=2)
        
        print("   • Cross-wavelet analysis...")
        spectral_analyzer.plot_cross_wavelet_analysis()
        
        print("   • Global wavelet spectra...")
        spectral_analyzer.plot_global_wavelet_spectra()
        
        # Export spectral results
        print("\n💾 EXPORTING SPECTRAL RESULTS...")
        
        spectral_analyzer.export_results('spectral_analysis_results.csv')
        
        spectral_pdf = spectral_analyzer.create_pdf_report(cycles_found)
        print(f"✅ Spectral analysis PDF report: {spectral_pdf}")
        
        spectral_json = spectral_analyzer.create_json_metadata(cycles_found)
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
        
        # Data loading (reuse same files)
        print("\n📂 LOADING DATA FOR LEAD-LAG ANALYSIS...")
        if not leadlag_analyzer.load_data(proxy1_file, proxy2_file):
            print("❌ Error in data loading for lead-lag analysis!")
            return
        
        # Data processing
        print("\n🔧 PROCESSING DATA FOR LEAD-LAG ANALYSIS...")
        leadlag_analyzer.interpolate_to_common_grid()
        
        # Lead-lag analysis
        print(f"\n🔄 LEAD-LAG ANALYSIS (max lag: {config.LEADLAG_ANALYSIS['max_lag_kyr']} kyr)...")
        leadlag_results = leadlag_analyzer.calculate_comprehensive_leadlag_analysis()
        
        # Generate lead-lag visualizations
        print("\n📈 GENERATING LEAD-LAG VISUALIZATIONS...")
        
        print("   • Comprehensive lead-lag analysis...")
        leadlag_analyzer.plot_comprehensive_leadlag_analysis()
        
        # Export lead-lag results
        print("\n💾 EXPORTING LEAD-LAG RESULTS...")
        
        leadlag_analyzer.export_results('leadlag_analysis_results.csv')
        
        leadlag_pdf = leadlag_analyzer.create_pdf_report()
        print(f"✅ Lead-lag analysis PDF report: {leadlag_pdf}")
        
        leadlag_json = leadlag_analyzer.create_json_metadata()
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