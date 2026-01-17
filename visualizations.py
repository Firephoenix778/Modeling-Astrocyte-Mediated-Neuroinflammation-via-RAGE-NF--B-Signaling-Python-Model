"""
Visualization Suite for ISEF Alzheimer's Model
==============================================
Creates publication-quality figures for competition presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for professional figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

def plot_main_timecourse(results, save_path='/home/claude/fig1_timecourse.png'):
    """
    Figure 1: Main time-course showing all key species
    This is your primary results figure for ISEF
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Astrocyte Inflammatory Response to Amyloid-β', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: RAGE Dynamics
    ax = axes[0, 0]
    ax.plot(results['Time_min'], results['RAGE'], 'b-', linewidth=2.5, label='Free RAGE')
    ax.plot(results['Time_min'], results['Ab_RAGE'], 'r-', linewidth=2.5, label='Aβ-RAGE Complex')
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Concentration (μM)', fontweight='bold')
    ax.set_title('A) RAGE Activation', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    ax.annotate('RAGE binding\nto Aβ', xy=(10, 0.03), xytext=(25, 0.04),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='left')
    
    # Panel B: NF-κB Signaling
    ax = axes[0, 1]
    ax.plot(results['Time_min'], results['NFkB_IkB'], '--', color='gray', 
            linewidth=2, label='NF-κB-IκB (Inactive)', alpha=0.7)
    ax.plot(results['Time_min'], results['IkB'], ':', color='orange',
            linewidth=2, label='Free IκB', alpha=0.7)
    ax.plot(results['Time_min'], results['NFkB_nuc'], '-', color='darkred', 
            linewidth=2.5, label='Nuclear NF-κB (Active)')
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Concentration (μM)', fontweight='bold')
    ax.set_title('B) NF-κB Activation', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Annotate key events
    peak_idx = results['NFkB_nuc'].idxmax()
    peak_time = results.loc[peak_idx, 'Time_min']
    peak_val = results.loc[peak_idx, 'NFkB_nuc']
    ax.annotate(f'Peak activation\n{peak_time:.1f} min', 
                xy=(peak_time, peak_val), xytext=(peak_time+10, peak_val+0.005),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                fontsize=9, ha='left')
    
    # Panel C: Inflammatory Outputs
    ax = axes[1, 0]
    ax2 = ax.twinx()
    
    l1 = ax.plot(results['Time_min'], results['IL6'], '-', color='purple', 
                 linewidth=2.5, label='IL-6')
    l2 = ax2.plot(results['Time_min'], results['ROS'], '-', color='darkorange', 
                  linewidth=2.5, label='ROS')
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('IL-6 Concentration (μM)', fontweight='bold', color='purple')
    ax2.set_ylabel('ROS Concentration (μM)', fontweight='bold', color='darkorange')
    ax.set_title('C) Inflammatory Mediators', fontweight='bold', loc='left')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax.grid(alpha=0.3)
    
    # Combined legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, frameon=True, loc='upper left')
    
    # Panel D: Apoptosis Marker
    ax = axes[1, 1]
    ax.plot(results['Time_min'], results['Casp3'] * 1000, '-', 
            color='darkgreen', linewidth=2.5, label='Caspase-3')
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Caspase-3 (nM)', fontweight='bold')
    ax.set_title('D) Neuronal Damage Marker', fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    
    # Annotate threshold crossing
    threshold_idx = np.where(results['Casp3'] > 0.01)[0]
    if len(threshold_idx) > 0:
        threshold_time = results.loc[threshold_idx[0], 'Time_min']
        ax.axvline(threshold_time, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(threshold_time + 2, ax.get_ylim()[1] * 0.7, 
                'Apoptotic\nthreshold', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_dose_response(dose_results, save_path='/home/claude/fig2_dose_response.png'):
    """
    Figure 2: Dose-response analysis showing effect of different Aβ concentrations
    Critical for showing model can predict biological responses
    NOW WITH CASPASE-3 PANEL!
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Dose-Response to Amyloid-β Concentration', 
                 fontsize=16, fontweight='bold')
    
    # Define color map
    doses = sorted([float(k.replace('uM', '')) for k in dose_results.keys()])
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(doses)))
    
    # Panel A: Peak responses vs dose
    ax = axes[0]
    peak_NFkB = []
    peak_IL6 = []
    peak_Casp3 = []
    
    for dose, color in zip(doses, colors):
        df = dose_results[f'{dose}uM']
        peak_NFkB.append(df['NFkB_nuc'].max())
        peak_IL6.append(df['IL6'].max())
        peak_Casp3.append(df['Casp3'].max())
    
    ax.plot(doses, peak_NFkB, 'o-', linewidth=2.5, markersize=8, 
            label='Peak NF-κB', color='darkred')
    ax.plot(doses, peak_IL6, 's-', linewidth=2.5, markersize=8, 
            label='Peak IL-6', color='purple')
    ax.plot(doses, peak_Casp3, '^-', linewidth=2.5, markersize=8, 
            label='Peak Caspase-3', color='darkgreen')
    
    ax.set_xlabel('Amyloid-β Concentration (μM)', fontweight='bold')
    ax.set_ylabel('Peak Concentration (μM)', fontweight='bold')
    ax.set_title('A) Peak Response vs. Aβ Dose', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='upper left')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, which='both')
    
    # Panel B: NF-κB time courses for different doses
    ax = axes[1]
    for dose, color in zip(doses, colors):
        df = dose_results[f'{dose}uM']
        ax.plot(df['Time_min'], df['NFkB_nuc'], '-', linewidth=2, 
                color=color, label=f'{dose} μM Aβ', alpha=0.8)
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Nuclear NF-κB (μM)', fontweight='bold')
    ax.set_title('B) NF-κB Time Course', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best', title='Aβ Dose', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel C: Caspase-3 time courses (NEW!)
    ax = axes[2]
    for dose, color in zip(doses, colors):
        df = dose_results[f'{dose}uM']
        ax.plot(df['Time_min'], df['Casp3'] * 1000, '-', linewidth=2, 
                color=color, label=f'{dose} μM Aβ', alpha=0.8)
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Caspase-3 (nM)', fontweight='bold')
    ax.set_title('C) Caspase-3 Time Course', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best', title='Aβ Dose', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add annotation showing dose-dependent effect
    final_casp3_values = [dose_results[f'{d}uM']['Casp3'].iloc[-1] * 1000 for d in doses]
    min_val = min(final_casp3_values)
    max_val = max(final_casp3_values)
    fold_change = max_val / min_val if min_val > 0 else 0
    
    if fold_change > 1.2:  # Only show if there's meaningful change
        ax.text(0.98, 0.02, f'Dose-dependent\n{fold_change:.1f}× increase',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_sensitivity_analysis(sensitivity_results, param_name, 
                              save_path='/home/claude/fig3_sensitivity.png'):
    """
    Figure 3: Parameter sensitivity analysis
    Shows which parameters most affect model outcomes - critical for ISEF judges
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', 
                 fontsize=16, fontweight='bold')
    
    fold_changes = sorted([float(k.replace('x', '')) for k in sensitivity_results.keys()])
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fold_changes)))
    
    # Panel A: NF-κB response
    ax = axes[0]
    for fc, color in zip(fold_changes, colors):
        df = sensitivity_results[f'{fc}x']
        ax.plot(df['Time_min'], df['NFkB_nuc'], '-', linewidth=2.5,
                color=color, label=f'{fc}× baseline', alpha=0.8)
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Nuclear NF-κB (μM)', fontweight='bold')
    ax.set_title('A) Effect on NF-κB Activation', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    
    # Panel B: IL-6 and Caspase-3 response
    ax = axes[1]
    for fc, color in zip(fold_changes, colors):
        df = sensitivity_results[f'{fc}x']
        ax.plot(df['Time_min'], df['IL6'], '-', linewidth=2,
                color=color, alpha=0.8)
    
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('IL-6 (μM)', fontweight='bold')
    ax.set_title('B) Effect on IL-6 Production', fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_intervention_comparison(control, intervention, intervention_name,
                                 save_path='/home/claude/fig4_intervention.png'):
    """
    Figure 4: Therapeutic intervention comparison
    Shows how model predicts drug effects - great for ISEF impact discussion
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Therapeutic Intervention: {intervention_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Panel A: NF-κB
    ax = axes[0, 0]
    ax.plot(control['Time_min'], control['NFkB_nuc'], '-', 
            linewidth=2.5, color='red', label='Control', alpha=0.8)
    ax.plot(intervention['Time_min'], intervention['NFkB_nuc'], '-',
            linewidth=2.5, color='blue', label=intervention_name, alpha=0.8)
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Nuclear NF-κB (μM)', fontweight='bold')
    ax.set_title('A) NF-κB Activation', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    
    # Calculate AUC for effect size
    control_auc = np.trapz(control['NFkB_nuc'], control['Time_min'])
    intervention_auc = np.trapz(intervention['NFkB_nuc'], intervention['Time_min'])
    reduction = (1 - intervention_auc / control_auc) * 100
    ax.text(0.98, 0.95, f'AUC Reduction:\n{reduction:.1f}%', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    # Panel B: IL-6
    ax = axes[0, 1]
    ax.plot(control['Time_min'], control['IL6'], '-',
            linewidth=2.5, color='red', label='Control', alpha=0.8)
    ax.plot(intervention['Time_min'], intervention['IL6'], '-',
            linewidth=2.5, color='blue', label=intervention_name, alpha=0.8)
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('IL-6 (μM)', fontweight='bold')
    ax.set_title('B) IL-6 Production', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    
    # Panel C: ROS
    ax = axes[1, 0]
    ax.plot(control['Time_min'], control['ROS'], '-',
            linewidth=2.5, color='red', label='Control', alpha=0.8)
    ax.plot(intervention['Time_min'], intervention['ROS'], '-',
            linewidth=2.5, color='blue', label=intervention_name, alpha=0.8)
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('ROS (μM)', fontweight='bold')
    ax.set_title('C) Oxidative Stress', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    
    # Panel D: Caspase-3
    ax = axes[1, 1]
    ax.plot(control['Time_min'], control['Casp3'] * 1000, '-',
            linewidth=2.5, color='red', label='Control', alpha=0.8)
    ax.plot(intervention['Time_min'], intervention['Casp3'] * 1000, '-',
            linewidth=2.5, color='blue', label=intervention_name, alpha=0.8)
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Caspase-3 (nM)', fontweight='bold')
    ax.set_title('D) Apoptosis Marker', fontweight='bold', loc='left')
    ax.legend(frameon=True, loc='best')
    ax.grid(alpha=0.3)
    
    # Calculate final reduction
    final_reduction = (1 - intervention['Casp3'].iloc[-1] / control['Casp3'].iloc[-1]) * 100
    ax.text(0.98, 0.95, f'Final Reduction:\n{final_reduction:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_summary_table(results, save_path='/home/claude/table1_metrics.csv'):
    """
    Create summary table of key metrics for paper/poster
    """
    metrics = {
        'Metric': [
            'Peak NF-κB (μM)',
            'Time to Peak NF-κB (min)',
            'Peak IL-6 (μM)',
            'Peak ROS (μM)',
            'Final Caspase-3 (nM)',
            'RAGE Depletion (%)',
            'NF-κB AUC (μM·min)'
        ],
        'Value': [
            results['NFkB_nuc'].max(),
            results.loc[results['NFkB_nuc'].idxmax(), 'Time_min'],
            results['IL6'].max(),
            results['ROS'].max(),
            results['Casp3'].iloc[-1] * 1000,
            (1 - results['RAGE'].min() / results['RAGE'].iloc[0]) * 100,
            np.trapz(results['NFkB_nuc'], results['Time_min'])
        ]
    }
    
    df = pd.DataFrame(metrics)
    df['Value'] = df['Value'].round(4)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved: {save_path}")
    
    return df


if __name__ == "__main__":
    # This script is meant to be imported, but can test basic functions
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_main_timecourse()")
    print("  - plot_dose_response()")
    print("  - plot_sensitivity_analysis()")
    print("  - plot_intervention_comparison()")
    print("  - create_summary_table()")
