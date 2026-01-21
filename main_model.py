import os
import numpy as np
import pickle
import healpy as hp
from tqdm import tqdm

from src.config import SATELLITES, SOLAR_THRESHOLDS, SEASONS, CLOCK_ANGLES, NSIDE, LAT_RES, LON_RES
from src.spha_model import SPHAModel

# --- Configuration ---
PROCESSED_DATA_PATH = "./data/processed/data_Ne(6)"
MODEL_OUTPUT_PATH = "./models"
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# Define which variables to model and their corresponding filename prefixes
# Key = Variable Name (for output file), Value = File Prefix (input file)
TARGET_VARIABLES = {
    'Ne': 'data_ne',
    'RODI': 'data_rodi',
    'Gamma': 'data_gamma'
}

def generate_model_coefficients(satellite='A', var_prefix='data_ne'):
    """
    Generates SPHA coefficients for a specific variable and satellite.
    """
    print(f"--- Processing {var_prefix} for Satellite {satellite} ---")
    
    # Initialize Model
    model = SPHAModel(nside=NSIDE)
    
    # Structure: alm_dict[SolarActivity][Season][ClockAngle]
    alm_dict = {}

    # Grid definitions
    lat_grid = np.arange(-90, 90, LAT_RES)
    lon_grid = np.arange(0, 360, LON_RES)
    
    # Iterate over conditions
    for sa in SOLAR_THRESHOLDS.keys():
        alm_dict[sa] = {}
        
        for season in SEASONS.keys():
            alm_dict[sa][season] = {}
            
            for ca in CLOCK_ANGLES.keys():
                
                # Dynamic Filename: e.g., data_rodi_Swarm_A_LSA_Summer_UR.npy
                filename = f"{var_prefix}_Swarm_{satellite}_{sa}_{season}_{ca}.npy"
                filepath = os.path.join(PROCESSED_DATA_PATH, filename)
                
                if not os.path.exists(filepath):
                    # Silent skip to avoid spamming logs, or print if debugging
                    # print(f"Missing: {filename}")
                    continue

                try:
                    # 1. Load Raw Data (Shape: 64800 x N_Obs)
                    data_grid_raw = np.load(filepath, allow_pickle=True)

                    # 2. Filter Outliers & Compress (Shape: 64800)
                    bin_means = model.apply_mad_filter(data_grid_raw)

                    # 3. Reshape to 2D Map (Shape: 180 x 360)
                    data_grid_2d = bin_means.reshape(len(lat_grid), len(lon_grid))
                    
                    # 4. Fill Gaps iteratively (Shape: 180 x 360)
                    data_grid_filled = model.fill_gaps_iterative(data_grid_2d)

                    # 5. Convert to Healpix
                    hp_map = model.grid_to_healpix(lat_grid, lon_grid, data_grid_filled)

                    # 6. Compute Coefficients
                    alm, cln = model.compute_coeffs(hp_map)
                    
                    alm_dict[sa][season][ca] = alm

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

    return alm_dict

def save_model(alm_dict, var_name, satellite):
    """Saves the coefficient dictionary to a unique pickle file."""
    filename = f"alm_dict_{var_name}_Swarm_{satellite}.pkl"
    path = os.path.join(MODEL_OUTPUT_PATH, filename)
    
    with open(path, 'wb') as f:
        pickle.dump(alm_dict, f)
    print(f"Saved model: {path}")

def main():
    # Loop through Satellites
    for sat in ['A']: # Add 'B' if you have data for Swarm B
        
        # Loop through Variables (Ne, RODI, Gamma)
        for var_name, file_prefix in TARGET_VARIABLES.items():
            
            # Generate
            alm_dict = generate_model_coefficients(satellite=sat, var_prefix=file_prefix)
            
            # Save
            if alm_dict: # Only save if we actually processed something
                save_model(alm_dict, var_name, sat)
            else:
                print(f"Warning: No data found for {var_name} on Swarm {sat}")

    print("\nAll models generated successfully.")

if __name__ == "__main__":
    main()
