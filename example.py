import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle
import pandas as pd
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

from src.spha_model import SPHAModel
from src.visualization import plot_polar_hemisphere
from src.config import NSIDE

# --- CONFIGURATION ---
MODEL_DIR = "./models"
SATELLITE = "A"

# Interpolation Reference Levels
SA_LEVELS = {'LSA': 70, 'MSA': 84, 'HSA': 133}
CA_LEVELS = {
    'UR': np.radians(30), 'R': np.radians(90), 'BR': np.radians(150),
    'B': np.radians(210), 'BL': np.radians(270), 'L': np.radians(330)
}

def get_user_inputs():
    print("\n" + "="*60)
    print("   Dynamic Model for Ionospheric Irregularities- Configuration")
    print("   (Press ENTER at any prompt to use the Default value)")
    print("="*60)
    
    # 1. Solar Activity
    sa_default = 120.0
    while True:
        try:
            sa_str = input(f">> Enter Solar Activity F10.7 [Range 60-140] (Default {sa_default}): ").strip()
            if sa_str == "":
                sa_input = sa_default
                print(f"   -> Using Default: {sa_input}")
                break
            val = float(sa_str)
            if 60.0 <= val <= 140.0:
                sa_input = val
                break
            else:
                print("   [!] Input out of range. Must be 60-140.")
        except ValueError:
            print("   [!] Invalid number.")

    # 2. Clock Angle
    ca_default = 90.0
    while True:
        try:
            ca_str = input(f">> Enter IMF Clock Angle [0-360] (Default {ca_default}°): ").strip()
            if ca_str == "":
                ca_deg = ca_default
                ca_input = np.radians(ca_deg)
                print(f"   -> Using Default: {ca_deg}°")
                break
            ca_deg = float(ca_str)
            ca_input = np.radians(ca_deg)
            break
        except ValueError:
            print("   [!] Invalid number.")

    # 3. Season
    season_default = 'Equinox'
    valid_seasons = ['Summer', 'Winter', 'Equinox']
    while True:
        season_str = input(f">> Enter Season {valid_seasons} (Default {season_default}): ").strip().capitalize()
        if season_str == "":
            season_input = season_default
            print(f"   -> Using Default: {season_input}")
            break
        if season_str in valid_seasons:
            season_input = season_str
            break
        else:
            print(f"   [!] Invalid season.")
        
    # 4. Custom L
    default_l = NSIDE * 2
    l_input = None
    while True:
        try:
            l_str = input(f">> Enter Max Harmonic Degree 'L' (Press Enter for default {default_l}): ").strip()
            if l_str == "":
                l_input = None
                print(f"   -> Using default L={default_l}")
                break
            val = int(l_str)
            if val > 0:
                l_input = val
                break
            else:
                print("   [!] L must be positive.")
        except ValueError:
            print("   [!] Invalid integer.")

    return sa_input, ca_input, season_input, l_input

def load_model(var_name):
    path = os.path.join(MODEL_DIR, f"alm_dict_{var_name}_Swarm_{SATELLITE}.pkl")
    if not os.path.exists(path):
        print(f"Error: Model file not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# --- CSV EXPORT FUNCTIONS ---
def healpix_to_grid(hp_map, lat_array, lon_array):
    """
    Inverse Mapping: Queries the Healpix map for values at specific Lat/Lon points.
    """
    rows = len(lat_array)
    cols = len(lon_array)
    grid_data = np.zeros((rows, cols))
    
    for i in range(rows):
        # Convert Lat to Theta (0 at North Pole)
        theta = np.radians(90.0 - lat_array[i])
        for j in range(cols):
            # Convert Lon to Phi
            phi = np.radians(lon_array[j])
            # Bilinear interpolation
            val = hp.get_interp_val(hp_map, theta, phi)
            grid_data[i, j] = val
    return grid_data

def save_csv_output(maps, season, sa, ca_rad):
    """
    Generates a regular 2x2 degree grid CSV file without redundant columns.
    """
    print("\n--- Generating CSV Output ---")
    ca_deg = int(np.degrees(ca_rad))
    
    # Define Target Grid (2x2 degree resolution)
    lat_north = np.arange(50, 92, 2)    # 50 to 90
    lat_south = np.arange(-50, -92, -2) # -50 to -90
    lon_grid = np.arange(0, 360, 2)     # 0 to 360
    
    # Extract Data (Inverse Mapping)
    print("   Sampling Northern Hemisphere...")
    ne_n = healpix_to_grid(maps['Ne'], lat_north, lon_grid)
    rodi_n = healpix_to_grid(maps['RODI'], lat_north, lon_grid)
    gamma_n = healpix_to_grid(maps['Gamma'], lat_north, lon_grid)

    print("   Sampling Southern Hemisphere...")
    ne_s = healpix_to_grid(maps['Ne'], lat_south, lon_grid)
    rodi_s = healpix_to_grid(maps['RODI'], lat_south, lon_grid)
    gamma_s = healpix_to_grid(maps['Gamma'], lat_south, lon_grid)
    
    data = []
    
    # Helper: Convert Longitude to MLT (0-360 -> 0-24)
    def get_mlt(lon): return lon / 15.0

    # Build Rows for North
    for i in range(len(lat_north)):
        for j in range(len(lon_grid)):
            data.append({
                'MLAT': lat_north[i],
                'MLT': get_mlt(lon_grid[j]),
                'Ne': ne_n[i, j],
                'RODI': rodi_n[i, j],
                'Gamma': gamma_n[i, j]
            })

    # Build Rows for South
    for i in range(len(lat_south)):
        for j in range(len(lon_grid)):
            data.append({
                'MLAT': lat_south[i],
                'MLT': get_mlt(lon_grid[j]),
                'Ne': ne_s[i, j],
                'RODI': rodi_s[i, j],
                'Gamma': gamma_s[i, j]
            })

    # Save
    filename = f"Ne_reconstructed_SA_{int(sa)}_{season}_IMFCA_{ca_deg}.csv"
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, float_format='%.4f')
    print(f"[SUCCESS] CSV Saved: {filename}")

def main():
    # 1. Get Inputs
    try:
        inputs = get_user_inputs()
        SOLAR_ACTIVITY_INPUT, CLOCK_ANGLE_INPUT, SEASON_INPUT, CUSTOM_L = inputs
    except KeyboardInterrupt:
        sys.exit(0)

    # 2. Load Models
    print("\n--- Loading Models (Ne, Gamma, RODI) ---")
    models = {
        'Ne': load_model('Ne'),
        'Gamma': load_model('Gamma'),
        'RODI': load_model('RODI')
    }
    
    if any(m is None for m in models.values()):
        print("Stopping due to missing models. Please run main_model.py first.")
        return

    print(f"--- Interpolating: {SEASON_INPUT}, F10.7={SOLAR_ACTIVITY_INPUT}, CA={np.degrees(CLOCK_ANGLE_INPUT):.0f}° ---")
    
    spha = SPHAModel(nside=NSIDE)
    maps = {}

    for var_name, alm_dict in models.items():
        interpolated_alm = spha.interpolate_coefficients(
            alm_dict, 
            sa_input=SOLAR_ACTIVITY_INPUT, 
            ca_input=CLOCK_ANGLE_INPUT, 
            season_input=SEASON_INPUT,
            sa_levels=SA_LEVELS,
            ca_levels=CA_LEVELS
        )
        maps[var_name] = spha.reconstruct_map(interpolated_alm, custom_lmax=CUSTOM_L)

    # 3. Plotting Setup
    print("\n--- Generating Visualization... ---")
    plt.figure(figsize=(24, 16))

    ne_data = maps['Ne'].copy()
    ne_data[ne_data <= 0] = np.nan 
    
    rodi_data = maps['RODI'].copy()
    rodi_data[rodi_data <= 0] = np.nan

    # ROW 1: NORTHERN HEMISPHERE
    plot_polar_hemisphere(
        np.log10(ne_data), hemisphere='North', sub=231, 
        vmin=4.5, vmax=np.log10(260000), unit_label='Log$_{10}$N$_e$ (cm$^{-3}$)'
    )
    plot_polar_hemisphere(
        maps['Gamma'], hemisphere='North', sub=232, 
        vmin=0.5, vmax=1.5, unit_label='$\\gamma(2)$',
        title=f'Northern Hemisphere (SA:{SOLAR_ACTIVITY_INPUT}sfu; CA:{np.degrees(CLOCK_ANGLE_INPUT)}°; {SEASON_INPUT})'
    )
    plot_polar_hemisphere(
        np.log10(rodi_data), hemisphere='North', sub=233, 
        vmin=np.log10(1000), vmax=np.log10(25000), unit_label='Log$_{10}$RODI'
    )

    # ROW 2: SOUTHERN HEMISPHERE
    plot_polar_hemisphere(
        np.log10(ne_data), hemisphere='South', sub=234, 
        vmin=4.5, vmax=np.log10(260000), unit_label='Log$_{10}$N$_e$ (cm$^{-3}$)',
    )
    plot_polar_hemisphere(
        maps['Gamma'], hemisphere='South', sub=235, 
        vmin=0.5, vmax=1.5, unit_label='$\\gamma(2)$',
        title=f'Southern Hemisphere (SA:{SOLAR_ACTIVITY_INPUT}; CA:{np.degrees(CLOCK_ANGLE_INPUT)}°; {SEASON_INPUT})'
    )
    plot_polar_hemisphere(
        np.log10(rodi_data), hemisphere='South', sub=236, 
        vmin=np.log10(1000), vmax=np.log10(25000), unit_label='Log$_{10}$RODI',
    )

    output_filename = "multi_parameter_view.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_filename}")
    
    # 4. Generate CSV (Independently)
    save_csv_output(maps, SEASON_INPUT, SOLAR_ACTIVITY_INPUT, CLOCK_ANGLE_INPUT)

    plt.show()

if __name__ == "__main__":
    main()
