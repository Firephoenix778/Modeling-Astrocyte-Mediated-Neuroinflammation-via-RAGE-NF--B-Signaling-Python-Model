import numpy as np
import pandas as pd
import os

from alzheimers_astrocyte_model_enhanced import AlzheimersAstrocyteModelEnhanced
from visualizations import (
    plot_main_timecourse, plot_dose_response,
    plot_sensitivity_analysis, plot_intervention_comparison,
    create_summary_table
)

# ============================================================
# Create output directories (Windows-safe)
# ============================================================
os.makedirs("output", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

FIG_DIR = "output/figures/"
TABLE_DIR = "output/tables/"
REPORT_PATH = "output/Enhanced_Model_Report.txt"

print("=" * 70)
print("ENHANCED MODEL - COMPLETE ANALYSIS (Windows-Compatible)")
print("=" * 70)
print()

# ============================================================================
# EXPERIMENT 1: BASELINE SIMULATION
# ============================================================================
print("EXPERIMENT 1: Baseline Simulation")
print("-" * 70)

model = AlzheimersAstrocyteModelEnhanced()
baseline = model.simulate(t_end=3600, n_points=2000)

print(f"✓ Simulation complete ({len(baseline)} time points)")
print()

# Create main figure
plot_main_timecourse(
    baseline,
    save_path=FIG_DIR + "Figure1_TimeCourse_Enhanced.png"
)

# Create summary table
summary = create_summary_table(
    baseline,
    save_path=TABLE_DIR + "Table1_Metrics_Enhanced.csv"
)

print("
Key Metrics:")
print(summary.to_string(index=False))
print()

# ============================================================================
# EXPERIMENT 2: DOSE-RESPONSE ANALYSIS
# ============================================================================
print("
" + "=" * 70)
print("EXPERIMENT 2: Dose-Response to Aβ")
print("-" * 70)

doses = [0.1, 1.0, 5.0, 10.0, 50.0]
dose_results = model.dose_response(doses, t_end=3600)
print("✓ Dose-response analysis complete")
print()

plot_dose_response(
    dose_results,
    save_path=FIG_DIR + "Figure2_DoseResponse_Enhanced.png"
)

# ============================================================================
# EXPERIMENT 3: PARAMETER SENSITIVITY
# ============================================================================
print("
" + "=" * 70)
print("EXPERIMENT 3: Parameter Sensitivity Analysis")
print("-" * 70)

critical_params = [
    'k_bind_RAGE',
    'k_IkB_deg',
    'k_NFkB_transloc',
    'k_IL6_transcription',
    'k_IkB_synthesis',
    'k_ROS_to_NFkB'
]

fold_changes = [0.1, 0.5, 1.0, 2.0, 10.0]

sensitivity_results = {}
print(f"Testing parameters: {critical_params}")
print()

for param in critical_params:
    results = model.parameter_sensitivity(param, fold_changes, t_end=3600)
    sensitivity_results[param] = results
    print(f"✓ Completed sensitivity for {param}")

# Save sensitivity plot for ROS feedback
plot_sensitivity_analysis(
    sensitivity_results['k_ROS_to_NFkB'],
    'k_ROS_to_NFkB',
    save_path=FIG_DIR + "Figure3_Sensitivity_ROS_Feedback.png"
)

# ============================================================================
# EXPERIMENT 4: THERAPEUTIC INTERVENTIONS
# ============================================================================
print("
" + "=" * 70)
print("EXPERIMENT 4: Therapeutic Interventions")
print("-" * 70)

control_model = AlzheimersAstrocyteModelEnhanced()
control = control_model.simulate(t_end=3600, n_points=2000)

interventions = []

# --- Intervention 1: RAGE inhibitor ---
model1 = AlzheimersAstrocyteModelEnhanced()
model1.k_bind_RAGE *= 0.1
intervention1 = model1.simulate(t_end=3600, n_points=2000)
interventions.append(("RAGE Inhibitor", intervention1))

# --- Intervention 2: NF-κB inhibitor ---
model2 = AlzheimersAstrocyteModelEnhanced()
model2.k_IkB_deg *= 0.2
intervention2 = model2.simulate(t_end=3600, n_points=2000)
interventions.append(("NF-κB Inhibitor", intervention2))

# --- Intervention 3: Anti-inflammatory ---
model3 = AlzheimersAstrocyteModelEnhanced()
model3.k_IkB_synthesis *= 5.0
intervention3 = model3.simulate(t_end=3600, n_points=2000)
interventions.append(("Anti-inflammatory", intervention3))

# --- Intervention 4: Antioxidant ---
model4 = AlzheimersAstrocyteModelEnhanced()
model4.k_ROS_scavenging *= 5.0
intervention4 = model4.simulate(t_end=3600, n_points=2000)
interventions.append(("Antioxidant", intervention4))

# Create top-2 intervention figures
for name, data in interventions[:2]:
    plot_intervention_comparison(
        control,
        data,
        name,
        save_path=FIG_DIR + f"Figure4_{name.replace(' ', '_')}_Enhanced.png"
    )

# ============================================================================
# EXPERIMENT 5: REPORT GENERATION
# ============================================================================
print("
" + "=" * 70)
print("GENERATING REPORT")
print("-" * 70)

with open(REPORT_PATH, "w") as f:
    f.write("ENHANCED MODEL REPORT (WINDOWS VERSION)
")
    f.write("=" * 70 + "

")
    f.write("Baseline peak values:
")
    f.write(f"Peak NF-κB: {baseline['NFkB_nuc'].max():.4f} μM
")
    f.write(f"Peak IL-6: {baseline['IL6'].max():.4f} μM
")
    f.write(f"Peak ROS: {baseline['ROS'].max():.4f} μM
")
    f.write(f"Final Casp3: {baseline['Casp3'].iloc[-1] * 1000:.2f} nM

")
    f.write("Figures saved in: output/figures/
")
    f.write("Tables saved in: output/tables/
")

print("✓ Report saved at:", REPORT_PATH)

print("
" + "=" * 70)
print("ANALYSIS COMPLETE — ALL FILES SAVED IN /output/")
print("=" * 70)
