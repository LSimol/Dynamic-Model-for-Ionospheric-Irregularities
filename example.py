import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle
import warnings
warnings.filterwarnings("ignore")

from src.spha_model import SPHAModel
from src.visualization import plot_polar_hemisphere
from src.config import NSIDE

# --- CONFIGURATION ---
MODEL_DIR = "./models"
SATELLITE = "A"

# Interpolation Inputs
SOLAR_ACTIVITY_INPUT = 127
CLOCK_ANGLE_INPUT = np.radians(139)
SEASON_INPUT = 'Equinox'

# Interpolation Reference Levels (from src/config.py or defined here)
SA_LEVELS = {'LSA': 70, 'MSA': 84, 'HSA': 133}
CA_LEVELS = {
    'UR': np.radians(30), 'R': np.radians(90), 'BR': np.radians(150),
    'B': np.radians(210), 'BL': np.radians(270), 'L': np.radians(330)
}

def load_model(var_name):
    """Helper to load a specific model file."""
    path = os.path.join(MODEL_DIR, f"alm_dict_{var_name}_Swarm_{SATELLITE}.pkl")
    if not os.path.exists(path):
        print(f"Error: Model file not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    # 1. Load All Models
    print("--- Loading Models (Ne, Gamma, RODI) ---")
    models = {
        'Ne': load_model('Ne'),
        'Gamma': load_model('Gamma'),
        'RODI': load_model('RODI')
    }
    
    # Check if all models loaded
    if any(m is None for m in models.values()):
        print("Stopping due to missing models. Please run main_model.py first.")
        return

    # 2. Interpolate & Reconstruct Maps
    print(f"--- Interpolating Conditions: SA={SOLAR_ACTIVITY_INPUT}, CA={np.degrees(CLOCK_ANGLE_INPUT):.0f}°, Season={SEASON_INPUT} ---")
    
    spha = SPHAModel(nside=NSIDE)
    maps = {}

    for var_name, alm_dict in models.items():
        # Interpolate Coefficients
        interpolated_alm = spha.interpolate_coefficients(
            alm_dict, 
            sa_input=SOLAR_ACTIVITY_INPUT, 
            ca_input=CLOCK_ANGLE_INPUT, 
            season_input=SEASON_INPUT,
            sa_levels=SA_LEVELS,
            ca_levels=CA_LEVELS
        )
        # Reconstruct Map
        maps[var_name] = spha.reconstruct_map(interpolated_alm)

    # 3. Plotting Setup
    # We want a grid: 2 Rows (North, South) x 3 Columns (Ne, Gamma, RODI)
    plt.figure(figsize=(24, 16)) # Adjusted size for 2x3 grid

    # --- ROW 1: NORTHERN HEMISPHERE ---
    # Ne (Log Scale)
    print("Plotting North Ne...")
    ne_data = maps['Ne'].copy()
    ne_data[ne_data <= 0] = np.nan # Safety for Log
    
    plot_polar_hemisphere(
        np.log10(ne_data), 
        hemisphere='North', 
        sub=(2, 3, 1), # Row 1, Col 1
        vmin=4.5, vmax=np.log10(260000), # 10^4.5 to 26*10^4
        unit_label='Log$_{10}$N$_e$ (cm$^{-3}$)',
        title=f'North Ne ({SEASON_INPUT})'
    )

    # Gamma (Linear Scale)
    print("Plotting North Gamma...")
    plot_polar_hemisphere(
        maps['Gamma'], 
        hemisphere='North', 
        sub=(2, 3, 2), # Row 1, Col 2
        vmin=0.5, vmax=1.5, 
        unit_label='$\\gamma(2)$',
        title=f'North Gamma ({SEASON_INPUT})'
    )

    # RODI (Log Scale)
    print("Plotting North RODI...")
    rodi_data = maps['RODI'].copy()
    rodi_data[rodi_data <= 0] = np.nan
    
    plot_polar_hemisphere(
        np.log10(rodi_data), 
        hemisphere='North', 
        sub=(2, 3, 3), # Row 1, Col 3
        vmin=np.log10(1000), vmax=np.log10(25000), 
        unit_label='Log$_{10}$RODI',
        title=f'North RODI ({SEASON_INPUT})'
    )

    # --- ROW 2: SOUTHERN HEMISPHERE ---
    # Ne (Log Scale)
    print("Plotting South Ne...")
    plot_polar_hemisphere(
        np.log10(ne_data), 
        hemisphere='South', 
        sub=(2, 3, 4), # Row 2, Col 1
        vmin=4.5, vmax=np.log10(260000),
        unit_label='Log$_{10}$N$_e$ (cm$^{-3}$)',
        title=f'South Ne'
    )

    # Gamma (Linear Scale)
    print("Plotting South Gamma...")
    plot_polar_hemisphere(
        maps['Gamma'], 
        hemisphere='South', 
        sub=(2, 3, 5), # Row 2, Col 2
        vmin=0.5, vmax=1.5,
        unit_label='$\\gamma(2)$',
        title=f'South Gamma'
    )

    # RODI (Log Scale)
    print("Plotting South RODI...")
    plot_polar_hemisphere(
        np.log10(rodi_data), 
        hemisphere='South', 
        sub=(2, 3, 6), # Row 2, Col 3
        vmin=np.log10(1000), vmax=np.log10(25000),
        unit_label='Log$_{10}$RODI',
        title=f'South RODI'
    )

    # Final Adjustments and Save
    output_filename = "multi_parameter_view.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Processing complete. Figure saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
