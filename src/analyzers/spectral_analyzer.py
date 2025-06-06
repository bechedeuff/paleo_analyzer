from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from scipy.ndimage import uniform_filter
import pycwt as wavelet
from pycwt.helpers import find

# Import configurations
from src import config

warnings.filterwarnings('ignore')

# Apply matplotlib and seaborn styling from configurations
plt.style.use(config.MATPLOTLIB_STYLE)
sns.set_palette(config.SEABORN_PALETTE)
plt.rcParams['figure.figsize'] = config.DEFAULT_FIGURE_SIZE
plt.rcParams['font.size'] = config.DEFAULT_FONT_SIZE


class PaleoclimateSpectralAnalyzer:
    """
    Class for spectral analysis of paleoclimate data using continuous wavelet transforms
    
    Main features:
    - Continuous Wavelet Transform (CWT) analysis
    - Cross-wavelet analysis and coherence
    - Global wavelet spectrum
    - Milankovitch cycle identification
    - Phase relationship analysis
    """
    
    def __init__(self, experiment_dir: Optional[str] = None) -> None:
        self.proxy1_data: Optional[pd.DataFrame] = None
        self.proxy2_data: Optional[pd.DataFrame] = None
        self.proxy1_name: Optional[str] = None
        self.proxy2_name: Optional[str] = None
        self.proxy1_units: Optional[str] = None
        self.proxy2_units: Optional[str] = None
        self.interpolated_data: Optional[pd.DataFrame] = None
        
        # Wavelet analysis results
        self.cwt1: Optional[np.ndarray] = None  # Complex wavelet transform
        self.cwt2: Optional[np.ndarray] = None
        self.cwt1_power: Optional[np.ndarray] = None
        self.cwt2_power: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
        self.freqs: Optional[np.ndarray] = None
        self.periods: Optional[np.ndarray] = None
        self.coi1: Optional[np.ndarray] = None  # Cone of influence
        self.coi2: Optional[np.ndarray] = None
        self.dt: Optional[float] = None  # Sampling interval
        
        # Data processing
        self.proxy1_normalized: Optional[np.ndarray] = None
        self.proxy2_normalized: Optional[np.ndarray] = None
        self.std1: Optional[float] = None
        self.std2: Optional[float] = None
        self.var1: Optional[float] = None
        self.var2: Optional[float] = None
        
        # Red noise characteristics
        self.alpha1: Optional[float] = None  # AR1 coefficient proxy 1
        self.alpha2: Optional[float] = None  # AR1 coefficient proxy 2
        
        # Cross-wavelet results
        self.cross_power: Optional[np.ndarray] = None
        self.coherence: Optional[np.ndarray] = None
        self.phase_difference: Optional[np.ndarray] = None
        
        # Significance testing
        self.significance1: Optional[np.ndarray] = None
        self.significance2: Optional[np.ndarray] = None
        self.sig95_1: Optional[np.ndarray] = None
        self.sig95_2: Optional[np.ndarray] = None
        self.global_signif1: Optional[np.ndarray] = None
        self.global_signif2: Optional[np.ndarray] = None
        
        # Global spectra
        self.global_power1: Optional[np.ndarray] = None
        self.global_power2: Optional[np.ndarray] = None
        
        self.experiment_dir: str = experiment_dir or 'results'
        
    def calculate_wavelet_transform(self, wavelet_type: str = config.SPECTRAL_ANALYSIS['wavelet_type'], 
                                    wavelet_param: float = config.SPECTRAL_ANALYSIS['wavelet_param']) -> None:
        """
        Calculate continuous wavelet transform for both proxies using PyCWT best practices
        
        Parameters:
        -----------
        wavelet_type : str
            Type of wavelet ('morlet', 'paul', 'dog')
        wavelet_param : float
            Wavelet parameter
        """
        if self.interpolated_data is None:
            raise ValueError("Execute interpolate_to_common_grid() first!")
            
        print(f"\n🔄 Calculating wavelet transforms using {wavelet_type} wavelet...")
        
        # Prepare time series data
        time = self.interpolated_data['age_kyr'].values
        proxy1 = self.interpolated_data['proxy1_values'].values
        proxy2 = self.interpolated_data['proxy2_values'].values
        
        # Sampling interval
        self.dt = np.mean(np.diff(time))
        
        # Detrend (remove linear trend)
        p1 = np.polyfit(time - time[0], proxy1, 1)
        proxy1_detrend = proxy1 - np.polyval(p1, time - time[0])
        
        p2 = np.polyfit(time - time[0], proxy2, 1)
        proxy2_detrend = proxy2 - np.polyval(p2, time - time[0])
        
        # Calculate statistics
        self.std1 = proxy1_detrend.std()
        self.std2 = proxy2_detrend.std()
        self.var1 = self.std1 ** 2
        self.var2 = self.std2 ** 2
        
        # Normalize
        self.proxy1_normalized = proxy1_detrend / self.std1
        self.proxy2_normalized = proxy2_detrend / self.std2
        
        # Calculate red noise parameters (AR1 autocorrelation)
        self.alpha1, _, _ = wavelet.ar1(proxy1_detrend)
        self.alpha2, _, _ = wavelet.ar1(proxy2_detrend)
        
        print(f"   AR1 coefficients: Proxy1={self.alpha1:.3f}, Proxy2={self.alpha2:.3f}")
        
        # Wavelet analysis parameters
        if wavelet_type.lower() == 'morlet':
            mother = wavelet.Morlet(wavelet_param)
        elif wavelet_type.lower() == 'paul':
            mother = wavelet.Paul(wavelet_param)
        elif wavelet_type.lower() == 'dog':
            mother = wavelet.DOG(wavelet_param)
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")
        
        # Starting scale and resolution
        s0 = 2 * self.dt  # Starting scale
        dj = 1 / 12  # Twelve sub-octaves per octave
        J = int(np.log2(config.SPECTRAL_ANALYSIS['max_period'] / (2 * self.dt)) / dj)  # Number of scales
        
        # Perform CWT for both proxies
        self.cwt1, self.scales, self.freqs, self.coi1, fft1, fftfreqs1 = wavelet.cwt(
            self.proxy1_normalized, self.dt, dj, s0, J, mother
        )
        
        self.cwt2, _, _, self.coi2, fft2, fftfreqs2 = wavelet.cwt(
            self.proxy2_normalized, self.dt, dj, s0, J, mother
        )
        
        # Calculate power spectra
        self.cwt1_power = np.abs(self.cwt1) ** 2
        self.cwt2_power = np.abs(self.cwt2) ** 2
        
        # Periods
        self.periods = 1 / self.freqs
        
        # Cross-wavelet analysis
        self.cross_power = self.cwt1 * np.conj(self.cwt2)
        self.phase_difference = np.angle(self.cross_power)
        
        # Calculate wavelet coherence
        self._calculate_wavelet_coherence()
        
        # Calculate significance levels
        self._calculate_significance_levels(mother)
        
        # Calculate global wavelet spectra
        self.global_power1 = self.cwt1_power.mean(axis=1)
        self.global_power2 = self.cwt2_power.mean(axis=1)
        
        # Calculate global significance
        N = len(self.proxy1_normalized)
        dof = N - self.scales  # Degrees of freedom correction
        
        self.global_signif1, _ = wavelet.significance(
            self.var1, self.dt, self.scales, 1, self.alpha1,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'], dof=dof, wavelet=mother
        )
        
        self.global_signif2, _ = wavelet.significance(
            self.var2, self.dt, self.scales, 1, self.alpha2,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'], dof=dof, wavelet=mother
        )
        
        print(f"✅ Wavelet analysis completed")
        print(f"   Period range: {self.periods.min():.1f} - {self.periods.max():.1f} kyr")
        print(f"   Frequency resolution: {len(self.periods)} bands")
        print(f"   Confidence level: {config.SPECTRAL_ANALYSIS['confidence_level']*100:.0f}%")
        
    def _calculate_significance_levels(self, mother) -> None:
        """
        Calculate significance levels for wavelet power spectra
        """
        N = len(self.proxy1_normalized)
        
        # Significance test for proxy 1
        self.significance1, _ = wavelet.significance(
            1.0, self.dt, self.scales, 0, self.alpha1,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'], wavelet=mother
        )
        
        # Significance test for proxy 2
        self.significance2, _ = wavelet.significance(
            1.0, self.dt, self.scales, 0, self.alpha2,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'], wavelet=mother
        )
        
        # Create significance arrays for plotting
        self.sig95_1 = np.ones([1, N]) * self.significance1[:, None]
        self.sig95_1 = self.cwt1_power / self.sig95_1
        
        self.sig95_2 = np.ones([1, N]) * self.significance2[:, None]
        self.sig95_2 = self.cwt2_power / self.sig95_2
        
    def _calculate_wavelet_coherence(self) -> None:
        """
        Calculate wavelet coherence using smoothing in time and scale domains
        """
        # Smooth in time and scale directions
        n_scales, n_times = self.cwt1_power.shape
        
        # Initialize coherence array
        self.coherence = np.zeros_like(self.cwt1_power, dtype=float)
        
        # For each scale, calculate coherence with appropriate smoothing
        for i, scale in enumerate(self.scales):
            # Time smoothing window (larger for larger scales)
            time_smooth = max(1, int(scale / self.dt / 6))  # 6 is a typical smoothing factor
            
            # Scale smoothing window
            scale_smooth = max(1, min(5, n_scales // 10))  # Usually 3-5 scales
            
            # Define smoothing windows around current scale
            scale_start = max(0, i - scale_smooth)
            scale_end = min(n_scales, i + scale_smooth + 1)
            
            # Extract windows for smoothing
            cross_window = self.cross_power[scale_start:scale_end, :]
            power1_window = self.cwt1_power[scale_start:scale_end, :]
            power2_window = self.cwt2_power[scale_start:scale_end, :]
            
            # Apply time smoothing using uniform filter
            if time_smooth > 1:
                # Smooth in time direction
                cross_smooth = uniform_filter(cross_window.real, size=(1, time_smooth), mode='nearest') + \
                              1j * uniform_filter(cross_window.imag, size=(1, time_smooth), mode='nearest')
                power1_smooth = uniform_filter(power1_window, size=(1, time_smooth), mode='nearest')
                power2_smooth = uniform_filter(power2_window, size=(1, time_smooth), mode='nearest')
            else:
                cross_smooth = cross_window
                power1_smooth = power1_window
                power2_smooth = power2_window
            
            # Apply scale smoothing (mean over scale window)
            cross_smooth_final = np.mean(cross_smooth, axis=0)
            power1_smooth_final = np.mean(power1_smooth, axis=0)
            power2_smooth_final = np.mean(power2_smooth, axis=0)
            
            # Calculate coherence for this scale
            # Avoid division by zero
            denominator = power1_smooth_final * power2_smooth_final
            valid_mask = denominator > 1e-12
            
            coherence_scale = np.zeros_like(cross_smooth_final, dtype=float)
            coherence_scale[valid_mask] = (np.abs(cross_smooth_final[valid_mask]) ** 2) / denominator[valid_mask]
            
            # Ensure coherence is between 0 and 1
            coherence_scale = np.clip(coherence_scale, 0, 1)
            
            self.coherence[i, :] = coherence_scale
        
        print(f"   ✅ Wavelet coherence calculated: {self.coherence.shape}")
        
    def calculate_scale_averaged_power(self, period_range: Tuple[float, float], mother) -> Tuple[np.ndarray, float]:
        """
        Calculate scale-averaged power for specified period range
        
        Parameters:
        -----------
        period_range : tuple
            (min_period, max_period) in kyr
        mother : wavelet object
            Mother wavelet for reconstruction
            
        Returns:
        --------
        scale_avg1, scale_avg2 : ndarray
            Scale-averaged power for both proxies
        scale_avg_signif : float
            Significance level for scale-averaged power
        """
        min_period, max_period = period_range
        
        # Find scales within the specified period range
        sel = find((self.periods >= min_period) & (self.periods < max_period))
        
        if len(sel) == 0:
            raise ValueError(f"No scales found in period range {min_period}-{max_period} kyr")
        
        # Calculate scale-averaged power (Torrence and Compo 1998, equation 24)
        Cdelta = mother.cdelta
        scale_avg = (self.scales * np.ones((len(self.proxy1_normalized), 1))).transpose()
        
        scale_avg1 = self.cwt1_power / scale_avg
        scale_avg1 = self.var1 * self.dt / Cdelta * scale_avg1[sel, :].sum(axis=0)
        
        scale_avg2 = self.cwt2_power / scale_avg
        scale_avg2 = self.var2 * self.dt / Cdelta * scale_avg2[sel, :].sum(axis=0)
        
        # Calculate significance level for scale-averaged power
        scale_avg_signif1, _ = wavelet.significance(
            self.var1, self.dt, self.scales, 2, self.alpha1,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'],
            dof=[self.scales[sel[0]], self.scales[sel[-1]]],
            wavelet=mother
        )
        
        scale_avg_signif2, _ = wavelet.significance(
            self.var2, self.dt, self.scales, 2, self.alpha2,
            significance_level=config.SPECTRAL_ANALYSIS['confidence_level'],
            dof=[self.scales[sel[0]], self.scales[sel[-1]]],
            wavelet=mother
        )
        
        return (scale_avg1, scale_avg2), (scale_avg_signif1, scale_avg_signif2)
        
    def identify_milankovitch_cycles(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Identify Milankovitch cycles in the wavelet spectra
        
        Returns:
        --------
        dict : Dictionary with identified cycles for each proxy
        """
        if self.cwt1_power is None:
            raise ValueError("Execute calculate_wavelet_transform() first!")
            
        print("\n🔍 Identifying Milankovitch cycles...")
        
        cycles_found = {
            'proxy1': [],
            'proxy2': [],
            'coherent': []
        }
        
        # Calculate global wavelet spectra (variance across time)
        global_power1 = np.var(self.cwt1_power, axis=1)
        global_power2 = np.var(self.cwt2_power, axis=1) 
        
        # Calculate global coherence if available
        if self.coherence is not None and self.coherence.size > 0:
            global_coherence = np.mean(self.coherence, axis=1)
        else:
            global_coherence = np.zeros_like(global_power1)
        
        # Check each Milankovitch cycle
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for target_period in cycle_periods:
                # Find closest period in our analysis
                period_idx = np.argmin(np.abs(self.periods - target_period))
                actual_period = self.periods[period_idx]
                
                if np.abs(actual_period - target_period) / target_period < 0.2:  # Within 20%
                    power1 = global_power1[period_idx]
                    power2 = global_power2[period_idx]
                    coherence = global_coherence[period_idx]
                    
                    # Check if power is significant (above 75th percentile)
                    if power1 > np.percentile(global_power1, 75):
                        cycles_found['proxy1'].append((actual_period, power1))
                        
                    if power2 > np.percentile(global_power2, 75):
                        cycles_found['proxy2'].append((actual_period, power2))
                        
                    if coherence > config.SPECTRAL_ANALYSIS['coherence_threshold']:
                        cycles_found['coherent'].append((actual_period, coherence))
        
        return cycles_found
        
    def plot_comprehensive_wavelet_analysis(self, proxy_num: int = 1,
                                            figsize: Tuple[int, int] = config.SPECTRAL_PLOTS['wavelet_power_figsize']) -> None:
        """
        Create comprehensive wavelet analysis plot following PyCWT tutorial style
        
        Parameters:
        -----------
        proxy_num : int
            Proxy number (1 or 2)
        figsize : tuple
            Figure size
        """
        if self.cwt1_power is None:
            raise ValueError("Execute calculate_wavelet_transform() first!")
            
        # Select proxy data
        if proxy_num == 1:
            power = self.cwt1_power
            proxy_name = self.proxy1_name
            proxy_data = self.interpolated_data['proxy1_values'].values
            proxy_norm = self.proxy1_normalized
            sig95 = self.sig95_1
            global_power = self.global_power1
            global_signif = self.global_signif1
            std = self.std1
            var = self.var1
            coi = self.coi1
        else:
            power = self.cwt2_power
            proxy_name = self.proxy2_name
            proxy_data = self.interpolated_data['proxy2_values'].values
            proxy_norm = self.proxy2_normalized
            sig95 = self.sig95_2
            global_power = self.global_power2
            global_signif = self.global_signif2
            std = self.std2
            var = self.var2
            coi = self.coi2
        
        time = self.interpolated_data['age_kyr'].values
        
        # Inverse wavelet transform for reconstruction
        iwave = wavelet.icwt(self.cwt1 if proxy_num == 1 else self.cwt2, 
                            self.scales, self.dt, 1/12, 
                            wavelet.Morlet(config.SPECTRAL_ANALYSIS['wavelet_param'])) * std
        
        # Create figure with PyCWT tutorial layout
        fig = plt.figure(figsize=figsize)
        
        # First subplot: Original and reconstructed time series
        ax1 = plt.axes([0.1, 0.75, 0.65, 0.2])
        ax1.plot(time, iwave, '-', linewidth=1, color='gray', alpha=0.7, label='Reconstructed')
        ax1.plot(time, proxy_data, 'k', linewidth=1.5, label='Original')
        ax1.set_title(f'a) {proxy_name} - Original and Wavelet Reconstruction')
        ax1.set_ylabel(f'{proxy_name}')
        ax1.legend(fontsize=10)
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: Wavelet power spectrum with significance and COI
        ax2 = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax1)
        
        # Power spectrum levels (log2 scale)
        levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        im = ax2.contourf(time, np.log2(self.periods), np.log2(power), 
                            np.log2(levels), extend='both', 
                            cmap=config.SPECTRAL_PLOTS['wavelet_power_colormap'])
        
        # Significance contours
        extent = [time.min(), time.max(), 0, max(self.periods)]
        ax2.contour(time, np.log2(self.periods), sig95, [-99, 1], 
                    colors='k', linewidths=2, extent=extent)
        
        # Cone of influence
        ax2.fill(np.concatenate([time, time[-1:] + self.dt, time[-1:] + self.dt,
                                time[:1] - self.dt, time[:1] - self.dt]),
                np.concatenate([np.log2(coi), [1e-9], np.log2(self.periods[-1:]),
                                np.log2(self.periods[-1:]), [1e-9]]),
                'k', alpha=0.3, hatch='x', label='COI')
        
        ax2.set_title(f'b) {proxy_name} Wavelet Power Spectrum ({config.SPECTRAL_ANALYSIS["wavelet_type"].title()})')
        ax2.set_ylabel('Period (kyr)')
        
        # Y-axis ticks in period scale
        Yticks = 2 ** np.arange(np.ceil(np.log2(self.periods.min())),
                                np.ceil(np.log2(self.periods.max())))
        ax2.set_yticks(np.log2(Yticks))
        ax2.set_yticklabels(Yticks.astype(int))
        ax2.invert_xaxis()
        
        # Add Milankovitch cycle indicators
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for period in cycle_periods:
                if self.periods.min() <= period <= self.periods.max():
                    ax2.axhline(y=np.log2(period), color='white', linestyle='--', alpha=0.7)
                    ax2.text(time.max() * 0.02, np.log2(period), f'{period}', 
                            color='white', fontsize=8, va='center')
        
        # Third subplot: Global wavelet spectrum
        ax3 = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=ax2)
        ax3.plot(global_signif, np.log2(self.periods), 'k--', label='95% confidence')
        ax3.plot(var * global_power, np.log2(self.periods), 'k-', linewidth=1.5, label='Global spectrum')
        ax3.set_title('c) Global Spectrum')
        ax3.set_xlabel(f'Power')
        ax3.set_xlim([0, (var * global_power).max() * 1.1])
        ax3.set_ylim(np.log2([self.periods.min(), self.periods.max()]))
        ax3.set_yticks(np.log2(Yticks))
        ax3.set_yticklabels([])
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Fourth subplot: Scale-averaged power for a specific band
        ax4 = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax1)
        
        # Calculate scale-averaged power for 2-8 kyr band (example)
        try:
            mother = wavelet.Morlet(config.SPECTRAL_ANALYSIS['wavelet_param'])
            (scale_avg1, scale_avg2), (scale_avg_signif1, scale_avg_signif2) = self.calculate_scale_averaged_power((2, 8), mother)
            
            if proxy_num == 1:
                scale_avg = scale_avg1
                scale_avg_signif = scale_avg_signif1
            else:
                scale_avg = scale_avg2
                scale_avg_signif = scale_avg_signif2
                
            ax4.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1, label='95% confidence')
            ax4.plot(time, scale_avg, 'k-', linewidth=1.5, label='Scale-averaged power')
            ax4.set_title('d) 2-8 kyr Scale-Averaged Power')
            ax4.legend(fontsize=8)
        except:
            ax4.text(0.5, 0.5, 'Scale-averaged power\nnot available for this period range', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('d) Scale-Averaged Power (unavailable)')
        
        ax4.set_xlabel('Age (kyr ago)')
        ax4.set_ylabel('Power')
        ax4.invert_xaxis()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/wavelet_analysis_proxy{proxy_num}.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Comprehensive wavelet analysis saved: {filename}")
        
    def plot_cross_wavelet_analysis(self, figsize: Tuple[int, int] = config.SPECTRAL_PLOTS['cross_wavelet_figsize']) -> None:
        """
        Plot cross-wavelet analysis including coherence and phase relationships
        """
        if self.cross_power is None:
            raise ValueError("Execute calculate_wavelet_transform() first!")
            
        if self.coherence is None or self.coherence.size == 0:
            print("⚠️  Coherence not calculated, skipping coherence plots")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        time = self.interpolated_data['age_kyr'].values
        
        # Cross-wavelet power
        cross_power_abs = np.abs(self.cross_power)
        im1 = ax1.contourf(time, self.periods, cross_power_abs, 
                            levels=20, cmap=config.SPECTRAL_PLOTS['cross_wavelet_colormap'])
        ax1.set_ylabel('Period (kyr)')
        ax1.set_title('Cross-Wavelet Power')
        ax1.set_yscale('log')
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        plt.colorbar(im1, ax=ax1, label='Power')
        
        # Coherence
        im2 = ax2.contourf(time, self.periods, self.coherence, 
                            levels=np.linspace(0, 1, 21), cmap='viridis')
        ax2.set_ylabel('Period (kyr)')
        ax2.set_title('Wavelet Coherence')
        ax2.set_yscale('log')
        ax2.invert_yaxis()
        ax2.invert_xaxis()
        
        # Add coherence threshold line
        coherent_regions = self.coherence > config.SPECTRAL_ANALYSIS['coherence_threshold']
        ax2.contour(time, self.periods, coherent_regions, levels=[0.5], 
                    colors='black', linewidths=2)
        
        plt.colorbar(im2, ax=ax2, label='Coherence')
        
        # Phase difference
        im3 = ax3.contourf(time, self.periods, self.phase_difference, 
                            levels=np.linspace(-np.pi, np.pi, 21), 
                            cmap='RdBu_r')
        ax3.set_xlabel('Age (kyr ago)')
        ax3.set_ylabel('Period (kyr)')
        ax3.set_title('Phase Difference')
        ax3.set_yscale('log')
        ax3.invert_yaxis()
        ax3.invert_xaxis()
        plt.colorbar(im3, ax=ax3, label='Phase (radians)')
        
        # Global coherence
        global_coherence = np.mean(self.coherence, axis=1)
        ax4.semilogx(global_coherence, self.periods, 'b-', linewidth=2, label='Global Coherence')
        ax4.axvline(x=config.SPECTRAL_ANALYSIS['coherence_threshold'], color='r', linestyle='--', 
                    label=f'Threshold ({config.SPECTRAL_ANALYSIS["coherence_threshold"]})')
        
        # Add Milankovitch cycles
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for period in cycle_periods:
                ax4.axhline(y=period, color='gray', linestyle=':', alpha=0.5)
                ax4.text(0.02, period, f'{period}', fontsize=8, va='center')
        
        ax4.set_xlabel('Coherence')
        ax4.set_ylabel('Period (kyr)')
        ax4.set_title('Global Coherence')
        ax4.set_ylim([self.periods.min(), self.periods.max()])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, 
                            wspace=0.25, hspace=0.3)
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/cross_wavelet_analysis.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Cross-wavelet analysis saved: {filename}")
        
    def plot_global_wavelet_spectra(self, figsize: Tuple[int, int] = config.SPECTRAL_PLOTS['global_spectrum_figsize']) -> None:
        """
        Plot global wavelet spectra for both proxies
        """
        if self.cwt1_power is None:
            raise ValueError("Execute calculate_wavelet_transform() first!")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate global spectra
        global_power1 = np.var(self.cwt1_power, axis=1)
        global_power2 = np.var(self.cwt2_power, axis=1)
        
        # Plot global spectra
        ax1.loglog(self.periods, global_power1, 'b-', linewidth=2, label=self.proxy1_name)
        ax1.loglog(self.periods, global_power2, 'r-', linewidth=2, label=self.proxy2_name)
        
        # Add Milankovitch cycles
        if config.SPECTRAL_PLOTS['global_spectrum_show_milankovitch']:
            for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
                for period in cycle_periods:
                    ax1.axvline(x=period, color='gray', linestyle='--', 
                                alpha=config.SPECTRAL_PLOTS['global_spectrum_milankovitch_alpha'])
                    ax1.text(period, global_power1.max() * 0.8, f'{period}', 
                            rotation=90, fontsize=8, ha='center')
        
        ax1.set_xlabel('Period (kyr)')
        ax1.set_ylabel('Power')
        ax1.set_title('Global Wavelet Spectra')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Coherence spectrum
        global_coherence = np.mean(self.coherence, axis=1)
        ax2.semilogx(self.periods, global_coherence, 'purple', linewidth=2, 
                        label='Global Coherence')
        ax2.axhline(y=config.SPECTRAL_ANALYSIS['coherence_threshold'], color='r', linestyle='--',
                    label=f'Threshold ({config.SPECTRAL_ANALYSIS["coherence_threshold"]})')
        
        # Add Milankovitch cycles
        if config.SPECTRAL_PLOTS['global_spectrum_show_milankovitch']:
            for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
                for period in cycle_periods:
                    ax2.axvline(x=period, color='gray', linestyle='--', 
                                alpha=config.SPECTRAL_PLOTS['global_spectrum_milankovitch_alpha'])
        
        ax2.set_xlabel('Period (kyr)')
        ax2.set_ylabel('Coherence')
        ax2.set_title('Global Coherence Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{self.experiment_dir}/figures/global_wavelet_spectra.png'
        plt.savefig(filename, dpi=config.DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        print(f"✅ Global wavelet spectra saved: {filename}")
        
    def _get_cycle_name(self, period: float) -> str:
        """
        Get the name of Milankovitch cycle based on period
        """
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for target_period in cycle_periods:
                if abs(period - target_period) / target_period < 0.2:
                    return cycle_name.replace('_', ' ').title()
        return 'Unknown'
