import numpy as np
from typing import Optional, Dict, Any

# Import all necessary modules
from src.analyzers.rolling_window_analyzer import PaleoclimateRollingWindowAnalyzer
from src.analyzers.spectral_analyzer import PaleoclimateSpectralAnalyzer
from src.analyzers.lead_lag_analyzer import PaleoclimateLead_LagAnalyzer

from src.utils.data_loader import load_paleoclimate_data, interpolate_to_common_grid
from src.utils.exporters import export_rolling_window_results, export_spectral_results, export_leadlag_results
from src.utils.file_utils import create_results_directory, save_experiment_config
from src.utils.validation import validate_configurations, print_configuration_summary

from src.reports.metadata import create_rolling_window_metadata, create_spectral_metadata, create_leadlag_metadata
from src.reports.pdf_reports import create_rolling_window_pdf_report, create_spectral_pdf_report, create_leadlag_pdf_report

from src import config

class PaleoclimateAnalysisPipeline:
    """
    Main orchestrator
    
    This class manages the workflow including:
    - Data loading and validation
    - Rolling window analysis
    - Spectral analysis
    - Lead-lag analysis
    - Results export and reporting
    """
    
    def __init__(self, proxy1_file: Optional[str] = None, proxy2_file: Optional[str] = None):
        """
        Initialize the analysis pipeline
        
        Parameters:
        -----------
        proxy1_file : str, optional
            Path to first proxy file. If None, uses config default
        proxy2_file : str, optional
            Path to second proxy file. If None, uses config default
        """
        self.proxy1_file = proxy1_file or f'data/{config.PROXY1_FILE}'
        self.proxy2_file = proxy2_file or f'data/{config.PROXY2_FILE}'
        self.experiment_dir = None
        self.data = None
        
    def setup_experiment(self) -> bool:
        """
        Setup experiment directory and validate configurations
        
        Returns:
        --------
        bool
            True if setup successful, False otherwise
        """
        print("🌍 PALEOCLIMATE ANALYSIS SUITE")
        print("=" * 70)
        print("Running complete paleoclimate analysis pipeline")
        print("Results will be organized in experiment subdirectories")
        print("-" * 70)
        
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
        self.experiment_dir = create_results_directory()
        
        # Save current configuration for reproducibility
        save_experiment_config(self.experiment_dir)
        
        return True
    
    def load_and_prepare_data(self) -> bool:
        """
        Load and prepare data for all analyses
        
        Returns:
        --------
        bool
            True if data loading successful, False otherwise
        """
        print("\n📂 LOADING AND PREPARING DATA...")
        success, self.data = load_paleoclimate_data(self.proxy1_file, self.proxy2_file)
        if not success or not self.data:
            print("❌ Error in data loading!")
            return False
        
        # Interpolate data to common grid (done once for all analyses)
        print("🔄 Interpolating to common grid...")
        self.data['interpolated_data'] = interpolate_to_common_grid(
            self.data['proxy1_data'], self.data['proxy2_data'])
        
        print("✅ Data preparation completed!")
        return True
    
    def run_rolling_window_analysis(self) -> bool:
        """
        Execute rolling window correlation analysis
        
        Returns:
        --------
        bool
            True if analysis successful, False otherwise
        """
        print("\n" + "="*70)
        print("🔄 ROLLING WINDOW CORRELATION ANALYSIS")
        print("="*70)
        
        try:
            # Initialize analyzer
            analyzer = PaleoclimateRollingWindowAnalyzer(f'{self.experiment_dir}/rolling_window')
            
            # Set analyzer attributes
            analyzer.proxy1_data = self.data['proxy1_data']
            analyzer.proxy2_data = self.data['proxy2_data']
            analyzer.proxy1_name = self.data['proxy1_name']
            analyzer.proxy2_name = self.data['proxy2_name']
            analyzer.proxy1_units = self.data['proxy1_units']
            analyzer.proxy2_units = self.data['proxy2_units']
            analyzer.interpolated_data = self.data['interpolated_data']
            
            # Run analysis
            print(f"\n📊 CALCULATING ROLLING CORRELATION (window: {config.ROLLING_WINDOW_ANALYSIS['window_size']} kyr)...")
            analyzer.calculate_rolling_correlation()
            
            print("\n🔍 IDENTIFYING CORRELATION PERIODS...")
            periods = analyzer.identify_correlation_periods()
            
            # Generate visualizations
            print("\n📈 GENERATING ROLLING WINDOW VISUALIZATIONS...")
            print("   • Comprehensive analysis...")
            analyzer.plot_comprehensive_analysis()
            
            print("   • Temporal evolution...")
            analyzer.plot_temporal_evolution()
            
            print("   • Window size comparison...")
            analyzer.compare_window_sizes()
            
            # Export results
            print("\n💾 EXPORTING ROLLING WINDOW RESULTS...")
            if analyzer.rolling_correlation is not None:
                export_rolling_window_results(analyzer.rolling_correlation, analyzer.proxy1_name, 
                                                analyzer.proxy2_name, analyzer.experiment_dir, 
                                                'rolling_correlation_results.csv')
            
            # Generate reports
            print("\n📄 Generating reports...")
            rolling_pdf = create_rolling_window_pdf_report(
                analyzer.rolling_correlation, analyzer.interpolated_data,
                analyzer.proxy1_name, analyzer.proxy2_name, 
                config.ROLLING_WINDOW_ANALYSIS['window_size'], periods, analyzer.experiment_dir
            )
            print(f"✅ Rolling window PDF report: {rolling_pdf}")
            
            rolling_json = create_rolling_window_metadata(analyzer.rolling_correlation, analyzer.proxy1_data, 
                                                            analyzer.proxy2_data, analyzer.interpolated_data,
                                                            analyzer.proxy1_name, analyzer.proxy2_name,
                                                            config.ROLLING_WINDOW_ANALYSIS['window_size'], periods, 
                                                            analyzer.experiment_dir)
            print(f"✅ Rolling window metadata JSON: {rolling_json}")
            
            # Summary
            corr_stats = analyzer.rolling_correlation['rolling_correlation']
            print(f"\n📊 ROLLING WINDOW SUMMARY:")
            print(f"   Mean correlation: {corr_stats.mean():.3f} (range: {corr_stats.min():.3f} to {corr_stats.max():.3f})")
            print(f"   Periods found: {len(periods['high_positive'])} pos | {len(periods['high_negative'])} neg | {len(periods['decoupled'])} decoupled")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in rolling window analysis: {e}")
            return False
    
    def run_spectral_analysis(self) -> bool:
        """
        Execute spectral analysis
        
        Returns:
        --------
        bool
            True if analysis successful, False otherwise
        """
        print("\n" + "="*70)
        print("🌊 SPECTRAL ANALYSIS")
        print("="*70)
        
        try:
            # Initialize analyzer
            analyzer = PaleoclimateSpectralAnalyzer(f'{self.experiment_dir}/spectral')
            
            # Set analyzer attributes (reuse loaded data)
            analyzer.proxy1_data = self.data['proxy1_data']
            analyzer.proxy2_data = self.data['proxy2_data']
            analyzer.proxy1_name = self.data['proxy1_name']
            analyzer.proxy2_name = self.data['proxy2_name']
            analyzer.proxy1_units = self.data['proxy1_units']
            analyzer.proxy2_units = self.data['proxy2_units']
            analyzer.interpolated_data = self.data['interpolated_data']
            
            # Wavelet analysis
            print(f"\n🌊 WAVELET ANALYSIS ({config.SPECTRAL_ANALYSIS['wavelet_type']} wavelet)...")
            analyzer.calculate_wavelet_transform()
            
            # Identify cycles
            print("\n🔍 IDENTIFYING MILANKOVITCH CYCLES...")
            cycles_found = analyzer.identify_milankovitch_cycles()
            
            # Generate visualizations
            print("\n📈 GENERATING SPECTRAL VISUALIZATIONS...")
            print("   • Wavelet analysis...")
            analyzer.plot_comprehensive_wavelet_analysis(proxy_num=1)
            analyzer.plot_comprehensive_wavelet_analysis(proxy_num=2)
            
            print("   • Cross-wavelet analysis...")
            analyzer.plot_cross_wavelet_analysis()
            
            print("   • Global wavelet spectra...")
            analyzer.plot_global_wavelet_spectra()
            
            # Export results
            print("\n💾 EXPORTING SPECTRAL RESULTS...")
            if analyzer.cwt1_power is not None:
                export_spectral_results(analyzer.cwt1_power, analyzer.cwt2_power, 
                                        analyzer.coherence, analyzer.periods, 
                                        analyzer.freqs, analyzer.proxy1_name, 
                                        analyzer.proxy2_name, analyzer.experiment_dir, 
                                        'spectral_analysis_results.csv')
            
            # Generate reports
            print("\n📄 Generating spectral reports...")
            spectral_pdf = create_spectral_pdf_report(
                analyzer.interpolated_data, analyzer.periods,
                analyzer.proxy1_name, analyzer.proxy2_name,
                cycles_found, analyzer.experiment_dir
            )
            print(f"✅ Spectral analysis PDF report: {spectral_pdf}")
            
            spectral_json = create_spectral_metadata(analyzer.interpolated_data, analyzer.cwt1_power, 
                                                    analyzer.cwt2_power, analyzer.coherence, 
                                                    analyzer.periods, analyzer.proxy1_name, 
                                                    analyzer.proxy2_name, cycles_found, 
                                                    analyzer.experiment_dir)
            print(f"✅ Spectral analysis metadata JSON: {spectral_json}")
            
            # Summary
            total_cycles = sum(len(cycles) for cycles in cycles_found.values())
            mean_coherence = (np.mean(analyzer.coherence) 
                                if analyzer.coherence is not None and analyzer.coherence.size > 0 
                                else 0)
            print(f"\n🌊 SPECTRAL ANALYSIS SUMMARY:")
            print(f"   Frequency bands analyzed: {len(analyzer.periods)}")
            print(f"   Milankovitch cycles found: {total_cycles}")
            print(f"   Mean coherence: {mean_coherence:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in spectral analysis: {e}")
            return False
    
    def run_leadlag_analysis(self) -> bool:
        """
        Execute lead-lag analysis
        
        Returns:
        --------
        bool
            True if analysis successful, False otherwise
        """
        print("\n" + "="*70)
        print("🔄 LEAD-LAG ANALYSIS")
        print("="*70)
        
        try:
            # Initialize analyzer
            analyzer = PaleoclimateLead_LagAnalyzer(f'{self.experiment_dir}/lead_lag')
            
            # Set analyzer attributes (reuse loaded data)
            analyzer.proxy1_data = self.data['proxy1_data']
            analyzer.proxy2_data = self.data['proxy2_data']
            analyzer.proxy1_name = self.data['proxy1_name']
            analyzer.proxy2_name = self.data['proxy2_name']
            analyzer.proxy1_units = self.data['proxy1_units']
            analyzer.proxy2_units = self.data['proxy2_units']
            analyzer.interpolated_data = self.data['interpolated_data']
            
            # Lead-lag analysis
            print(f"\n🔄 LEAD-LAG ANALYSIS (max lag: {config.LEADLAG_ANALYSIS['max_lag_kyr']} kyr)...")
            leadlag_results = analyzer.calculate_comprehensive_leadlag_analysis()
            
            # Generate visualizations
            print("\n📈 GENERATING LEAD-LAG VISUALIZATIONS...")
            print("   • Lead-lag analysis...")
            analyzer.plot_comprehensive_leadlag_analysis()
            
            # Export results
            print("\n💾 EXPORTING LEAD-LAG RESULTS...")
            if analyzer.leadlag_results is not None:
                export_leadlag_results(analyzer.leadlag_results, analyzer.experiment_dir, 
                                        'leadlag_analysis_results.csv')
            
            # Generate reports
            print("\n📄 Generating lead-lag reports...")
            leadlag_pdf = create_leadlag_pdf_report(
                analyzer.leadlag_results, analyzer.interpolated_data,
                analyzer.proxy1_name, analyzer.proxy2_name,
                analyzer.experiment_dir
            )
            print(f"✅ Lead-lag analysis PDF report: {leadlag_pdf}")
            
            leadlag_json = create_leadlag_metadata(analyzer.leadlag_results, analyzer.interpolated_data,
                                                    analyzer.proxy1_name, analyzer.proxy2_name, 
                                                    analyzer.experiment_dir)
            print(f"✅ Lead-lag analysis metadata JSON: {leadlag_json}")
            
            # Summary
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
            
            return True
            
        except Exception as e:
            print(f"❌ Error in lead-lag analysis: {e}")
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete analysis pipeline
        
        Returns:
        --------
        bool
            True if all analyses successful, False otherwise
        """
        try:
            # Setup
            if not self.setup_experiment():
                return False
            
            # Data loading
            if not self.load_and_prepare_data():
                return False
            
            # Run all analyses
            rolling_success = self.run_rolling_window_analysis()
            spectral_success = self.run_spectral_analysis()
            leadlag_success = self.run_leadlag_analysis()
            
            # Final summary
            print("\n" + "="*70)
            print("🎉 ANALYSIS PIPELINE COMPLETED")
            print("="*70)
            print(f"📁 Results directory: {self.experiment_dir}")
            print(f"✅ Rolling Window Analysis: {'Success' if rolling_success else 'Failed'}")
            print(f"✅ Spectral Analysis: {'Success' if spectral_success else 'Failed'}")
            print(f"✅ Lead-Lag Analysis: {'Success' if leadlag_success else 'Failed'}")
            print("-" * 70)
            
            return rolling_success and spectral_success and leadlag_success
            
        except ImportError as e:
            print(f"❌ Error importing modules: {e}")
            print("Make sure all required packages are installed:")
            print("   pip install pandas numpy matplotlib seaborn scipy reportlab pycwt")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

def run_complete_analysis() -> None:
    """
    Function to run all the analyses
    """
    pipeline = PaleoclimateAnalysisPipeline()
    pipeline.run_complete_analysis() 