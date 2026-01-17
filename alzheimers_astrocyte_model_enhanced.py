"""
ENHANCED ISEF-Level Alzheimer's Astrocyte Model
================================================
Modeling Astrocyte-Mediated Neuroinflammation via RAGE-NF-κB Signaling
WITH ROS FEEDBACK TO NF-κB

This enhanced version includes ROS→NF-κB crosstalk, where oxidative stress
amplifies inflammatory signaling through IKK activation.

Author: ISEF Research Project (Enhanced Version)
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns

# Set publication-quality plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class AlzheimersAstrocyteModelEnhanced:
    """
    Enhanced mechanistic model of astrocyte inflammatory response to Aβ
    
    NEW FEATURE: ROS feedback to NF-κB activation
    
    Pathway:
    Aβ → RAGE → IκB degradation → NF-κB activation → 
    IL-6 production + IκB negative feedback → Caspase-3 activation
    
    PLUS: ROS amplifies IκB degradation (positive feedback loop)
    """
    
    def __init__(self):
        """Initialize model parameters based on literature"""
        
        # === BINDING PARAMETERS ===
        # RAGE-Aβ interaction (Yan et al., 1996, Nature)
        self.k_bind_RAGE = 1.0e-3      # μM⁻¹s⁻¹ (scaled for μM units)
        self.k_unbind_RAGE = 0.01      # s⁻¹ (Kd ~10 μM, typical for RAGE-Aβ)
        
        # === NF-κB SIGNALING PARAMETERS ===
        # IκB degradation by RAGE-activated IKK (Hoffmann et al., 2002, Science)
        self.k_IkB_deg = 0.005         # s⁻¹ (half-life ~2-3 min when phosphorylated)
        self.k_IkB_deg_basal = 0.0001  # s⁻¹ (slow basal degradation)
        
        # NEW: ROS feedback to NF-κB pathway
        self.k_ROS_to_NFkB = 0.002     # s⁻¹ (ROS oxidizes IKK, amplifying IκB degradation)
        
        # NF-κB-IκB complex formation/dissociation
        self.k_IkB_bind_NFkB = 0.1     # μM⁻¹s⁻¹
        self.k_IkB_unbind_NFkB = 0.001 # s⁻¹
        
        # NF-κB nuclear translocation
        self.k_NFkB_transloc = 0.05    # s⁻¹ (fast nuclear import)
        self.k_NFkB_export = 0.01      # s⁻¹ (slower export)
        
        # === GENE EXPRESSION PARAMETERS ===
        # IL-6 transcription (Lin et al., 2012, J Immunol)
        self.k_IL6_transcription = 0.0002  # s⁻¹ (gene expression is slower)
        self.k_IL6_degradation = 0.0001    # s⁻¹ (cytokine half-life ~hours)
        
        # IκB resynthesis (negative feedback) (Hoffmann et al., 2002)
        self.k_IkB_synthesis = 0.0003      # s⁻¹ (induced by nuclear NF-κB)
        self.k_IkB_synthesis_basal = 0.00005  # s⁻¹ (low basal synthesis)
        
        # === ROS PARAMETERS ===
        # RAGE generates oxidative stress (Lue et al., 2001)
        self.k_ROS_production = 0.001      # s⁻¹
        self.k_ROS_scavenging = 0.0005     # s⁻¹ (antioxidant systems)
        
        # === APOPTOSIS PARAMETERS ===
        # Caspase-3 activation (Green & Llambi, 2015, Cell)
        self.k_casp3_activation = 0.0005   # μM⁻²s⁻¹ (requires IL-6 + ROS)
        self.k_casp3_autoactivation = 10.0 # s⁻¹ (strong positive feedback)
        self.k_casp3_degradation = 0.0001  # s⁻¹
        
        # Threshold for caspase autoactivation
        self.casp3_threshold = 0.005       # μM
        
        # === INITIAL CONDITIONS (μM) ===
        self.initial_conditions = {
            'Ab': 0.1,           # Aβ starts low, will increase
            'RAGE': 0.05,        # ~30,000 receptors/cell
            'Ab_RAGE': 0.0,      # No complex initially
            'IkB': 0.1,          # Abundant IκB
            'NFkB_IkB': 0.15,    # Most NF-κB is sequestered
            'NFkB_cyto': 0.01,   # Small free cytoplasmic pool
            'NFkB_nuc': 0.001,   # Minimal basal nuclear NF-κB
            'IL6': 0.0,          # No IL-6 initially
            'ROS': 0.01,         # Basal ROS level
            'Casp3': 0.001       # Low basal caspase-3
        }
        
        # Store parameter names for sensitivity analysis
        self.param_names = [
            'k_bind_RAGE', 'k_unbind_RAGE', 'k_IkB_deg', 
            'k_NFkB_transloc', 'k_IL6_transcription', 
            'k_IkB_synthesis', 'k_casp3_activation',
            'k_ROS_to_NFkB'  # NEW parameter
        ]
    
    def Ab_stimulation(self, t):
        """
        Aβ stimulation protocol: ramp up to pathological levels
        
        In AD, Aβ accumulates gradually, then plateaus
        This simulates acute exposure for computational efficiency
        """
        if t < 100:
            return 0.1 + (10.0 - 0.1) * (t / 100)  # Ramp to 10 μM
        else:
            return 10.0  # Maintain at pathological level
    
    def derivatives(self, y, t):
        """
        System of ODEs describing the pathway WITH ROS FEEDBACK
        
        Parameters:
        -----------
        y : array
            Current state [Ab, RAGE, Ab_RAGE, IkB, NFkB_IkB, NFkB_cyto, NFkB_nuc, IL6, ROS, Casp3]
        t : float
            Current time (seconds)
        
        Returns:
        --------
        dydt : array
            Derivatives for each species
        """
        # Unpack current state
        Ab, RAGE, Ab_RAGE, IkB, NFkB_IkB, NFkB_cyto, NFkB_nuc, IL6, ROS, Casp3 = y
        
        # Apply Aβ stimulation
        Ab = self.Ab_stimulation(t)
        
        # === REACTION RATES ===
        
        # R1: Aβ + RAGE ⇌ Aβ-RAGE (binding equilibrium)
        r_RAGE_binding = self.k_bind_RAGE * Ab * RAGE
        r_RAGE_unbinding = self.k_unbind_RAGE * Ab_RAGE
        
        # R2: Aβ-RAGE → IκB degradation (RAGE signals to IKK)
        # ENHANCED: ROS now amplifies this reaction!
        ROS_contribution = self.k_ROS_to_NFkB * ROS
        r_IkB_degradation = ((self.k_IkB_deg * Ab_RAGE + 
                             self.k_IkB_deg_basal + 
                             ROS_contribution) * IkB)
        
        # R3: NF-κB-IκB → NF-κB + IκB_degraded (IκB proteolysis releases NF-κB)
        # ENHANCED: ROS amplifies this too!
        r_NFkB_release = ((self.k_IkB_deg * Ab_RAGE + 
                          self.k_IkB_deg_basal + 
                          ROS_contribution) * NFkB_IkB)
        
        # R4: NF-κB + IκB → NF-κB-IκB (complex reformation)
        r_NFkB_sequestration = self.k_IkB_bind_NFkB * NFkB_cyto * IkB
        
        # R5: NF-κB_cyto → NF-κB_nuc (nuclear translocation)
        r_NFkB_import = self.k_NFkB_transloc * NFkB_cyto
        r_NFkB_export = self.k_NFkB_export * NFkB_nuc
        
        # R6: NF-κB_nuc → IL-6 (transcription)
        r_IL6_production = self.k_IL6_transcription * NFkB_nuc
        r_IL6_degradation = self.k_IL6_degradation * IL6
        
        # R7: NF-κB_nuc → IκB (negative feedback)
        r_IkB_resynthesis = (self.k_IkB_synthesis * NFkB_nuc + 
                             self.k_IkB_synthesis_basal)
        
        # R8: Aβ-RAGE → ROS (oxidative stress)
        r_ROS_production = self.k_ROS_production * Ab_RAGE
        r_ROS_scavenging = self.k_ROS_scavenging * ROS
        
        # R9: IL-6 + ROS → Caspase-3 activation (combined inflammatory stress)
        r_Casp3_activation = self.k_casp3_activation * IL6 * ROS
        
        # R10: Caspase-3 autoactivation (switch-like behavior)
        if Casp3 > self.casp3_threshold:
            r_Casp3_auto = self.k_casp3_autoactivation * Casp3
        else:
            r_Casp3_auto = 0
        
        r_Casp3_degradation = self.k_casp3_degradation * Casp3
        
        # === DIFFERENTIAL EQUATIONS ===
        
        dAb_dt = 0  # Aβ is externally controlled
        
        dRAGE_dt = -r_RAGE_binding + r_RAGE_unbinding
        
        dAb_RAGE_dt = r_RAGE_binding - r_RAGE_unbinding
        
        dIkB_dt = (-r_IkB_degradation - r_NFkB_sequestration + 
                   r_IkB_resynthesis)
        
        dNFkB_IkB_dt = (r_NFkB_sequestration - r_NFkB_release)
        
        dNFkB_cyto_dt = (r_NFkB_release - r_NFkB_sequestration - 
                         r_NFkB_import + r_NFkB_export)
        
        dNFkB_nuc_dt = r_NFkB_import - r_NFkB_export
        
        dIL6_dt = r_IL6_production - r_IL6_degradation
        
        dROS_dt = r_ROS_production - r_ROS_scavenging
        
        dCasp3_dt = (r_Casp3_activation + r_Casp3_auto - 
                     r_Casp3_degradation)
        
        return [dAb_dt, dRAGE_dt, dAb_RAGE_dt, dIkB_dt, dNFkB_IkB_dt, 
                dNFkB_cyto_dt, dNFkB_nuc_dt, dIL6_dt, dROS_dt, dCasp3_dt]
    
    def simulate(self, t_end=3600, n_points=1000):
        """
        Run time-course simulation
        
        Parameters:
        -----------
        t_end : float
            Simulation end time (seconds, default 1 hour)
        n_points : int
            Number of time points
        
        Returns:
        --------
        results : DataFrame
            Time-course data for all species
        """
        # Time vector
        t = np.linspace(0, t_end, n_points)
        
        # Initial conditions as array
        y0 = list(self.initial_conditions.values())
        
        # Solve ODEs
        solution = odeint(self.derivatives, y0, t)
        
        # Create DataFrame
        species_names = list(self.initial_conditions.keys())
        results = pd.DataFrame(solution, columns=species_names)
        results['Time_s'] = t
        results['Time_min'] = t / 60
        
        return results
    
    def parameter_sensitivity(self, param_name, fold_changes, t_end=3600):
        """
        Analyze sensitivity to a specific parameter
        
        Parameters:
        -----------
        param_name : str
            Name of parameter to vary
        fold_changes : array-like
            Fold changes to test (e.g., [0.1, 1.0, 10.0])
        t_end : float
            Simulation time
        
        Returns:
        --------
        results : dict
            Dictionary of DataFrames for each fold change
        """
        original_value = getattr(self, param_name)
        results = {}
        
        for fc in fold_changes:
            # Modify parameter
            setattr(self, param_name, original_value * fc)
            
            # Run simulation
            df = self.simulate(t_end=t_end)
            results[f'{fc}x'] = df
            
            # Restore original
            setattr(self, param_name, original_value)
        
        return results
    
    def dose_response(self, Ab_doses, t_end=3600):
        """
        Simulate response to different Aβ concentrations
        
        Parameters:
        -----------
        Ab_doses : array-like
            Aβ concentrations to test (μM)
        t_end : float
            Simulation time
        
        Returns:
        --------
        results : dict
            Dictionary of DataFrames for each dose
        """
        results = {}
        
        for dose in Ab_doses:
            # Modify Aβ stimulation
            original_stim = self.Ab_stimulation
            self.Ab_stimulation = lambda t: dose
            
            # Run simulation
            df = self.simulate(t_end=t_end)
            results[f'{dose}uM'] = df
            
            # Restore original
            self.Ab_stimulation = original_stim
        
        return results


def main():
    """Run basic simulation and create visualizations"""
    
    print("=" * 60)
    print("ENHANCED ISEF Alzheimer's Astrocyte Model")
    print("With ROS → NF-κB Feedback")
    print("=" * 60)
    print()
    
    # Initialize model
    model = AlzheimersAstrocyteModelEnhanced()
    
    # Run simulation
    print("Running baseline simulation (1 hour)...")
    results = model.simulate(t_end=3600, n_points=2000)
    
    print(f"✓ Simulation complete")
    print(f"  Time points: {len(results)}")
    print(f"  Duration: {results['Time_min'].max():.1f} minutes")
    print()
    
    # Display key metrics
    print("Key Metrics:")
    print("-" * 40)
    
    # Peak NF-κB activation
    peak_NFkB = results['NFkB_nuc'].max()
    peak_NFkB_time = results.loc[results['NFkB_nuc'].idxmax(), 'Time_min']
    print(f"Peak NF-κB (nuclear): {peak_NFkB:.4f} μM at {peak_NFkB_time:.1f} min")
    
    # Peak IL-6
    peak_IL6 = results['IL6'].max()
    peak_IL6_time = results.loc[results['IL6'].idxmax(), 'Time_min']
    print(f"Peak IL-6: {peak_IL6:.4f} μM at {peak_IL6_time:.1f} min")
    
    # Peak ROS
    peak_ROS = results['ROS'].max()
    peak_ROS_time = results.loc[results['ROS'].idxmax(), 'Time_min']
    print(f"Peak ROS: {peak_ROS:.4f} μM at {peak_ROS_time:.1f} min")
    
    # Final Caspase-3
    final_Casp3 = results['Casp3'].iloc[-1]
    print(f"Final Caspase-3: {final_Casp3:.4f} μM")
    
    # RAGE depletion
    RAGE_depletion = (1 - results['RAGE'].min() / model.initial_conditions['RAGE']) * 100
    print(f"RAGE depletion: {RAGE_depletion:.1f}%")
    print()
    
    # Save results
    results.to_csv('/home/claude/baseline_simulation_enhanced.csv', index=False)
    print("✓ Results saved to: baseline_simulation_enhanced.csv")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
