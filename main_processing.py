import os
import glob
from tqdm import tqdm
from src.config import SATELLITES, YEARS, MONTHS
from src.preprocessor import load_and_prepare_dataframe, calculate_gamma_for_file

# Paths (Update these)
RAW_DATA_PATH = "./data/raw" 
OUTPUT_PATH = "./data/processed"

def run_pipeline():
    for sat in SATELLITES:
        for year in YEARS:
            for month in MONTHS:
                print(f"Processing Sat {sat} - {year}/{month}...")
                
                # Construct path (adapt to your actual file naming convention)
                file_name = f"Swarm_LP_1Hz_{sat}_{year}_{month}_RODI10s.txt"
                file_path = os.path.join(RAW_DATA_PATH, str(year), file_name)
                
                if not os.path.exists(file_path):
                    continue

                # 1. Load
                df = load_and_prepare_dataframe(
                    file_path, 
                    usecols=[0, 1, 2, 3, 4, 5, 7, 10, 15, 17],
                    col_names=['Year', 'Month', 'Day', 'Hour', 'minute', 'second', 'DOY', 'Mlt', 'Mlat', 'Ne']
                )

                # 2. Calc Gamma
                df = calculate_gamma_for_file(df)

                # 3. Save Intermediate
                save_file = os.path.join(OUTPUT_PATH, f"Swarm_{sat}_{year}_{month}.json")
                df.to_json(save_file)

if __name__ == "__main__":
    run_pipeline()
