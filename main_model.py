import os
import numpy as np
import pickle
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import SATELLITES, SOLAR_THRESHOLDS, SEASONS, CLOCK_ANGLES, NSIDE, LAT_RES, LON_RES
from src.spha_model import SPHAModel

# --- Configuration ---
PROCESSED_DATA_PATH = "./data/processed"
MODEL_OUTPUT_PATH = "./models"
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# Define reference values for interpolation labels (from spha_total.ipynb)
# These represent the "center" values for the linear interpolation
SA_LEVELS = {'LSA': 70, 'MSA': 84, 'HSA': 133}
CA_LEVELS = {
    'UR': np.radians(30), 'R': np.radians(90), 'BR': np.radians(150),
    'B': np.radians(210), 'BL': np.radians(270), 'L': np.radians(330)
}

def generate_model_coefficients(satellite='A'):
    """
    Loads processed lat/lon grids, computes SPHA coefficients, 
    and returns a nested dictionary of coefficients.
    """
    print(f"--- Generating SPHA Model for Satellite {satellite} ---")
    
    # Initialize Model
    model = SPHAModel(nside=NSIDE)
    
    # Structure: alm_dict[SolarActivity][Season][ClockAngle]
    alm_dict = {}

    # Grid definitions (matching processing step)
    lat_grid = np.arange(-90, 90, LAT_RES)
    lon_grid = np.arange(0, 360, LON_RES)
    
    # Flatten grid coordinates for mapping to Healpix
    # Note: Logic must match how data was saved in processing step
    # Assuming data is saved as (lat_len * lon_len) 1D arrays or similar
    # We will reconstruct the 2D grid logic here for clarity
    
    # Iterating through all conditions
    for sa in SOLAR_THRESHOLDS.keys():
        alm_dict[sa] = {}
        
        for season in SEASONS.keys():
            alm_dict[sa][season] = {}
            
            for ca in CLOCK_ANGLES.keys():
                # Construct filename pattern (Adjust to match your actual save format)
                # Example from notebook: data_ne_Swarm_A_LSA_Summer_UR.npy
                filename = f"data_ne_Swarm_{satellite}_{sa}_{season}_{ca}.npy"
                filepath = os.path.join(PROCESSED_DATA_PATH, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: Missing file {filename}")
                    continue

                # 1. Load Data
                # Data is expected to be a 2D grid (lat x lon) after MAD filtering
                try:
                    data_grid = np.load(filepath)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue

                # 2. Convert Lat/Lon Grid to Healpix Map
                # We assume data_grid shape is (180, 360) corresponding to lat_grid, lon_grid
                hp_map = model.grid_to_healpix(lat_grid, lon_grid, data_grid)

                # 3. Compute SPHA Coefficients
                alm, cln = model.compute_coeffs(hp_map)
                
                # 4. Store Coefficients
                alm_dict[sa][season][ca] = alm

    return alm_dict

def save_model(alm_dict, filename="spha_coefficients.pkl"):
    """Saves the coefficient dictionary to a pickle file."""
    path = os.path.join(MODEL_OUTPUT_PATH, filename)
    with open(path, 'wb') as f:
        pickle.dump(alm_dict, f)
    print(f"Model saved to {path}")

def main():
    # 1. Generate Model for Swarm A (Example)
    alm_dict_A = generate_model_coefficients(satellite='A')
    
    # 2. Save the trained model
    save_model(alm_dict_A, filename="alm_dict_Ne_Swarm_A.pkl")
    
    # ---------------------------------------------------------
    # Example: Using the Model for Reconstruction (Interpolation)
    # ---------------------------------------------------------
    print("\n--- Running Interpolation Example ---")
    model = SPHAModel(nside=NSIDE)
    
    # Test Conditions
    test_sa = 84            # MSA
    test_ca = np.radians(30)# UR
    test_season = 'Summer'
    
    # Perform Interpolation
    try:
        interpolated_alm = model.interpolate_coefficients(
            alm_dict_A, 
            sa_input=test_sa, 
            ca_input=test_ca, 
            season_input=test_season,
            sa_levels=SA_LEVELS,
            ca_levels=CA_LEVELS
        )
        
        # Reconstruct Map
        reconstructed_map = model.reconstruct_map(interpolated_alm)
        
        # Plotting (Simple preview)
        hp.mollview(reconstructed_map, title=f"Reconstructed Ne ({test_season}, SA={test_sa}, CA={np.degrees(test_ca):.0f}°)")
        plt.savefig(os.path.join(MODEL_OUTPUT_PATH, "example_reconstruction.png"))
        print("Example reconstruction saved to models/example_reconstruction.png")
        
    except Exception as e:
        print(f"Interpolation failed (likely missing data for example keys): {e}")

if __name__ == "__main__":
    main()
