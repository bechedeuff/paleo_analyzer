from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Import configurations
from . import configurations as config

def create_rolling_window_pdf_report(rolling_correlation: pd.DataFrame, interpolated_data: pd.DataFrame,
                                        proxy1_name: str, proxy2_name: str, window_size: int,
                                        periods: Dict[str, pd.DataFrame], experiment_dir: str) -> str:
    """
    Create a well-formatted PDF report for rolling window analysis
    
    Parameters:
    -----------
    rolling_correlation : pd.DataFrame
        Rolling correlation results
    interpolated_data : pd.DataFrame
        Interpolated data
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    window_size : int
        Window size used in the analysis
    periods : dict
        Dictionary with classified periods
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    # Prepare data
    corr_stats = rolling_correlation['rolling_correlation']
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # PDF configuration
    pdf_file = f'{experiment_dir}/correlation_analysis_report.pdf'
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
        ['Proxy 1:', proxy1_name],
        ['Proxy 2:', proxy2_name],
        ['Window used:', f'{window_size} kyr'],
        ['Analyzed period:', f"{interpolated_data['age_kyr'].min():.1f} - {interpolated_data['age_kyr'].max():.1f} kyr ago"],
        ['Total points:', str(len(rolling_correlation))]
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
    <b>• POSITIVE CORRELATION:</b> {proxy1_name} and {proxy2_name} vary together
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

def create_spectral_pdf_report(interpolated_data: pd.DataFrame, periods: np.ndarray,
                                proxy1_name: str, proxy2_name: str,
                                cycles_found: Dict[str, List[Tuple[float, float]]],
                                experiment_dir: str) -> str:
    """
    Create a comprehensive PDF report for spectral analysis
    
    Parameters:
    -----------
    interpolated_data : pd.DataFrame
        Interpolated data
    periods : np.ndarray
        Periods array
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    cycles_found : dict
        Dictionary with identified cycles
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    def _get_cycle_name(period: float) -> str:
        """Get the name of Milankovitch cycle based on period"""
        for cycle_name, cycle_periods in config.SPECTRAL_ANALYSIS['milankovitch_cycles'].items():
            for target_period in cycle_periods:
                if abs(period - target_period) / target_period < 0.2:
                    return cycle_name.replace('_', ' ').title()
        return 'Unknown'
    
    # Prepare data
    analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # PDF configuration
    pdf_file = f'{experiment_dir}/spectral_analysis_report.pdf'
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
    
    # Main title
    story.append(Paragraph("SPECTRAL ANALYSIS REPORT - PALEOCLIMATE DATA", title_style))
    story.append(Spacer(1, 20))
    
    # General information
    info_data = [
        ['Analysis date:', analysis_date],
        ['Proxy 1:', proxy1_name],
        ['Proxy 2:', proxy2_name],
        ['Wavelet type:', config.SPECTRAL_ANALYSIS['wavelet_type']],
        ['Period range:', f"{periods.min():.1f} - {periods.max():.1f} kyr"],
        ['Coherence threshold:', str(config.SPECTRAL_ANALYSIS['coherence_threshold'])],
        ['Total time points:', str(len(interpolated_data))]
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
    
    # Milankovitch cycles found
    story.append(Paragraph("IDENTIFIED MILANKOVITCH CYCLES", section_style))
    
    cycle_data = [['Cycle Type', 'Period (kyr)', 'Power', 'Proxy']]
    
    for proxy_name, cycles in cycles_found.items():
        for period, power in cycles:
            cycle_data.append([
                _get_cycle_name(period),
                f'{period:.1f}',
                f'{power:.3f}',
                proxy_name
            ])
    
    if len(cycle_data) > 1:
        cycle_table = Table(cycle_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
        cycle_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
        ]))
        story.append(cycle_table)
    else:
        story.append(Paragraph("No significant Milankovitch cycles identified.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return pdf_file

def create_leadlag_pdf_report(leadlag_results: Dict[str, Any], interpolated_data: pd.DataFrame,
                                proxy1_name: str, proxy2_name: str, experiment_dir: str) -> str:
    """
    Create a well-formatted PDF report for lead-lag analysis
    
    Parameters:
    -----------
    leadlag_results : dict
        Lead-lag analysis results
    interpolated_data : pd.DataFrame
        Interpolated data
    proxy1_name : str
        Name of first proxy
    proxy2_name : str
        Name of second proxy
    experiment_dir : str
        Experiment directory path
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    # PDF configuration
    pdf_file = f'{experiment_dir}/leadlag_analysis_report.pdf'
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
        ['Proxy 1:', proxy1_name],
        ['Proxy 2:', proxy2_name],
        ['Analysis methods:', ', '.join(leadlag_results['analysis_info']['methods'])],
        ['Correlation types:', ', '.join(leadlag_results['analysis_info']['correlation_types'])],
        ['Max lag tested:', f"{leadlag_results['analysis_info']['max_lag_kyr']} kyr"],
        ['Total data points:', str(len(interpolated_data))]
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
    
    for method, method_data in leadlag_results['summary'].items():
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