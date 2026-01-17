import os
import numpy as np
import pandas as pd

from alzheimers_astrocyte_model_enhanced import AlzheimersAstrocyteModelEnhanced
from visualizations import (
    plot_main_timecourse,
    plot_dose_response,
    plot_sensitivity_analysis,
    plot_intervention_comparison,
    create_summary_table,
)

# ---------------------------------------------------------------------
# Set up output folders (works on Windows, Mac, Linux)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUTPUT_DIR, "Enhanced_Model_Report.txt")

print("=" * 70)
print("ENHANCED ALZHEIMER'S ASTROCYTE MODEL - FULL ISEF PIPELINE")
print("=" * 70)
print()

# =====================================================================
# EXPERIMENT 1: BASELINE SIMULATION
# =====================================================================
print("EXPERIMENT 1: Baseline Simulation")
print("-" * 70)

model = AlzheimersAstrocyteModelEnhanced()
baseline = model.simulate(t_end=3600, n_points=2000)

print(f"✓ Simulation complete ({len(baseline)} time points)")
print(f"  Duration: {baseline['Time_min'].max():.1f} minutes")
print()

# Main time-course figure
fig1_path = os.path.join(FIG_DIR, "Figure1_TimeCourse_Enhanced.png")
plot_main_timecourse(baseline, save_path=fig1_path)

# Summary metrics table
table1_path = os.path.join(TABLE_DIR, "Table1_Metrics_Enhanced.csv")
summary = create_summary_table(baseline, save_path=table1_path)

print("Key Metrics:")
print(summary.to_string(index=False))
print()

# =====================================================================
# EXPERIMENT 2: DOSE-RESPONSE ANALYSIS
# =====================================================================
print("=" * 70)
print("EXPERIMENT 2: Dose-Response to Aβ")
print("-" * 70)

doses = [0.1, 1.0, 5.0, 10.0, 50.0]
dose_results = model.dose_response(doses, t_end=3600)

print(f"Tested Aβ doses (μM): {doses}")
print("✓ Dose-response simulations complete")
print()

print("Dose-Response Summary (Peak Values):")
print("-" * 50)
for dose in doses:
    df = dose_results[f"{dose}uM"]
    peak_NFkB = df["NFkB_nuc"].max()
    peak_IL6 = df["IL6"].max()
    final_casp3 = df["Casp3"].iloc[-1] * 1000.0  # convert to nM for readability
    print(
        f"{dose:6.1f} μM Aβ → "
        f"Peak NF-κB: {peak_NFkB:.4f} μM, "
        f"Peak IL-6: {peak_IL6:.4f} μM, "
        f"Final Casp3: {final_casp3:.2f} nM"
    )
print()

fig2_path = os.path.join(FIG_DIR, "Figure2_DoseResponse_Enhanced.png")
plot_dose_response(dose_results, save_path=fig2_path)

# =====================================================================
# EXPERIMENT 3: PARAMETER SENSITIVITY ANALYSIS
# =====================================================================
print("=" * 70)
print("EXPERIMENT 3: Parameter Sensitivity Analysis")
print("-" * 70)

critical_params = [
    "k_bind_RAGE",
    "k_IkB_deg",
    "k_NFkB_transloc",
    "k_IL6_transcription",
    "k_IkB_synthesis",
    "k_ROS_to_NFkB",  # new ROS feedback parameter
]

fold_changes = [0.1, 0.5, 1.0, 2.0, 10.0]

sensitivity_results = {}

print(f"Parameters to test: {', '.join(critical_params)}")
print(f"Fold changes: {fold_changes}")
print()

for param in critical_params:
    print(f"  Analyzing sensitivity for {param} ...")
    results = model.parameter_sensitivity(param, fold_changes, t_end=3600)
    sensitivity_results[param] = results

    # Simple sensitivity index: compare AUC at 10x vs 1x, normalized
    baseline_auc = np.trapz(
        results["1.0x"]["NFkB_nuc"], results["1.0x"]["Time_min"]
    )
    high_auc = np.trapz(
        results["10.0x"]["NFkB_nuc"], results["10.0x"]["Time_min"]
    )
    if baseline_auc > 0:
        sens_index = (high_auc / baseline_auc) / 10.0
    else:
        sens_index = float("nan")
    print(f"    Sensitivity index (NF-κB AUC): {sens_index:.3f}")
print()

# Make a dedicated plot for the ROS feedback parameter
fig3_path = os.path.join(FIG_DIR, "Figure3_Sensitivity_ROS_Feedback.png")
plot_sensitivity_analysis(
    sensitivity_results["k_ROS_to_NFkB"],
    "k_ROS_to_NFkB",
    save_path=fig3_path,
)

# =====================================================================
# EXPERIMENT 4: THERAPEUTIC INTERVENTIONS
# =====================================================================
print("=" * 70)
print("EXPERIMENT 4: Therapeutic Interventions")
print("-" * 70)

# Fresh control simulation for fair comparison
control_model = AlzheimersAstrocyteModelEnhanced()
control = control_model.simulate(t_end=3600, n_points=2000)

interventions = []

# 1) RAGE Inhibitor: blocks Aβ–RAGE binding
print()
print("1. RAGE Inhibitor (e.g., FPS-ZM1)")
m1 = AlzheimersAstrocyteModelEnhanced()
m1.k_bind_RAGE *= 0.1
int1 = m1.simulate(t_end=3600, n_points=2000)
red1 = (1.0 - int1["Casp3"].iloc[-1] / control["Casp3"].iloc[-1]) * 100.0
print(f"   → Caspase-3 reduction: {red1:.1f}%")
interventions.append(("RAGE Inhibitor", int1, red1))

# 2) NF-κB Inhibitor: prevents IκB degradation
print()
print("2. NF-κB Inhibitor (e.g., BAY 11-7082)")
m2 = AlzheimersAstrocyteModelEnhanced()
m2.k_IkB_deg *= 0.2
int2 = m2.simulate(t_end=3600, n_points=2000)
red2 = (1.0 - int2["Casp3"].iloc[-1] / control["Casp3"].iloc[-1]) * 100.0
print(f"   → Caspase-3 reduction: {red2:.1f}%")
interventions.append(("NF-κB Inhibitor", int2, red2))

# 3) Anti-inflammatory: boosts IκB resynthesis
print()
print("3. Anti-inflammatory (e.g., Dexamethasone-like effect)")
m3 = AlzheimersAstrocyteModelEnhanced()
m3.k_IkB_synthesis *= 5.0
int3 = m3.simulate(t_end=3600, n_points=2000)
red3 = (1.0 - int3["Casp3"].iloc[-1] / control["Casp3"].iloc[-1]) * 100.0
print(f"   → Caspase-3 reduction: {red3:.1f}%")
interventions.append(("Anti-inflammatory", int3, red3))

# 4) Antioxidant: stronger ROS scavenging
print()
print("4. Antioxidant (e.g., N-acetylcysteine)")
m4 = AlzheimersAstrocyteModelEnhanced()
m4.k_ROS_scavenging *= 5.0
int4 = m4.simulate(t_end=3600, n_points=2000)
red4 = (1.0 - int4["Casp3"].iloc[-1] / control["Casp3"].iloc[-1]) * 100.0
nfkb_red4 = (1.0 - int4["NFkB_nuc"].max() / control["NFkB_nuc"].max()) * 100.0
il6_red4 = (1.0 - int4["IL6"].max() / control["IL6"].max()) * 100.0
print(f"   → Caspase-3 reduction: {red4:.1f}%")
print(f"   → NF-κB peak reduction: {nfkb_red4:.1f}%")
print(f"   → IL-6 peak reduction:  {il6_red4:.1f}%")
interventions.append(("Antioxidant", int4, red4))

print()
print("✓ Intervention simulations complete")
print()

# Rank interventions by efficacy (Casp3 reduction)
interventions.sort(key=lambda tup: tup[2], reverse=True)

print("Intervention Ranking (by Caspase-3 reduction):")
print("-" * 60)
for rank, (name, _, reduction) in enumerate(interventions, start=1):
    print(f"{rank}. {name:20s} → {reduction:5.1f}% reduction")
print()

# Plot comparison for top 2 interventions
for idx, (name, data, _) in enumerate(interventions[:2], start=1):
    fname = f"Figure4_{idx}_Intervention_{name.replace(' ', '_')}_Enhanced.png"
    fpath = os.path.join(FIG_DIR, fname)
    plot_intervention_comparison(control, data, name, save_path=fpath)

# =====================================================================
# EXPERIMENT 5: SUMMARIZE ROS FEEDBACK IMPACT
# =====================================================================
print("=" * 70)
print("EXPERIMENT 5: ROS → NF-κB Feedback Summary")
print("-" * 70)

print("Control (with ROS feedback):")
print(f"  Peak NF-κB: {control['NFkB_nuc'].max():.4f} μM")
print(f"  Peak IL-6:  {control['IL6'].max():.4f} μM")
print(f"  Final Casp3: {control['Casp3'].iloc[-1] * 1000.0:.2f} nM")
print()
print("Antioxidant (with ROS feedback):")
print(f"  Peak NF-κB: {int4['NFkB_nuc'].max():.4f} μM ({nfkb_red4:+.1f}%)")
print(f"  Peak IL-6:  {int4['IL6'].max():.4f} μM ({il6_red4:+.1f}%)")
print(f"  Final Casp3: {int4['Casp3'].iloc[-1] * 1000.0:.2f} nM ({red4:+.1f}%)")
print()

# =====================================================================
# COMPREHENSIVE TEXT REPORT
# =====================================================================
print("=" * 70)
print("GENERATING COMPREHENSIVE REPORT")
print("-" * 70)

lines = []
lines.append("=" * 70)
lines.append("ENHANCED MODEL REPORT - WITH ROS FEEDBACK (FULL PIPELINE)")
lines.append("=" * 70)
lines.append("")

lines.append("MODEL FEATURES:")
lines.append("- RAGE–NF-κB–IL-6 axis in astrocytes")
lines.append("- Caspase-3 as apoptosis / damage marker")
lines.append("- ROS → NF-κB positive feedback loop")
lines.append("")

lines.append("BASELINE KEY METRICS:")
lines.append(f"  Peak NF-κB: {baseline['NFkB_nuc'].max():.4f} μM")
lines.append(f"  Peak IL-6:  {baseline['IL6'].max():.4f} μM")
lines.append(f"  Peak ROS:   {baseline['ROS'].max():.4f} μM")
lines.append(f"  Final Casp3: {baseline['Casp3'].iloc[-1] * 1000.0:.2f} nM")
lines.append("")

lines.append("THERAPEUTIC INTERVENTION RANKING (by Casp3 reduction):")
for rank, (name, _, reduction) in enumerate(interventions, start=1):
    lines.append(f"  {rank}. {name:20s} → {reduction:5.1f}% reduction")
lines.append("")

lines.append("ANTIOXIDANT EFFECTS (ROS FEEDBACK IN ACTION):")
lines.append(f"  NF-κB peak reduction: {nfkb_red4:.1f}%")
lines.append(f"  IL-6 peak reduction:  {il6_red4:.1f}%")
lines.append(f"  Casp3 reduction:      {red4:.1f}%")
lines.append("")
lines.append("Files generated:")
lines.append(f"  - {os.path.basename(fig1_path)}")
lines.append(f"  - {os.path.basename(fig2_path)}")
lines.append(f"  - {os.path.basename(fig3_path)}")
lines.append("  - Two Figure4_* intervention comparison plots")
lines.append(f"  - {os.path.basename(table1_path)}")
lines.append(f"  - {os.path.basename(REPORT_PATH)}")
lines.append("")
lines.append("End of report.")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✓ Report written to: {REPORT_PATH}")
print()
print("=" * 70)
print("FULL ENHANCED ANALYSIS COMPLETE")
print("=" * 70)
