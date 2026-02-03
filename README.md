# Dynamic-Model-for-Ionospheric-Irregularities

This Python tool implements a dynamic model to characterize **ionospheric irregularities and turbulence** in the topside ionosphere (~460 km altitude) at high latitudes ($>|50^\circ$| Magnetic Latitude). Developed using 10 years of data from the ESA Swarm mission (2014–2023), the model provides a climatological description of the ionosphere's state for different magnetic latitudes, magnetic local times, seasons, solar activity and interplanetary magnetic field (IMF) orientations.

It specifically models three key parameters:
1) **Electron density ($N_e$)**: The background plasma density;
2) **Rate of Change of Density Index (RODI)**: Quantitfy **electron density variability** and proxy for the intensity of irregularities;
3) **Second Order Scaling Exponent ($\gamma$(2))**: serves as a proxy for ionospheric **turbulent regimes**.

Further details regarding the model and its performance can be found in **Mestici et al., 2026 (Accepted)**. Citation details will be updated upon publication.

## Methodology 
   
The model utilizes **Spherical Harmonic Analysis** to reconstruct spatial maps of $N_e$, RODI, and $\gamma(2)$ in a Magnetic Local Time (MLT) vs. Magnetic Latitude (MLat) reference frame.

**Input Data**: High-resolution (1 Hz) electron density measurements from the Swarm Langmuir Probes.

**Conditioning**: Data is binned by:

i) Solar Activity (F10.7 index): Low, Medium, High;

ii) Local Season: Summer, Winter, Equinox;

iii) IMF Clock Angle: 6 sectors ($0^\circ-360^\circ$).

**Interpolation**: The core innovation of this tool is its ability to linearly interpolate between the spherical harmonic coefficients of adjacent states. This allows the user to generate maps for any specific F10.7 value or Clock Angle, providing a continuous dynamic description rather than just static snapshots..

## Usage Guide

1. **Installation**
Clone the repository and install the required dependencies:

git clone https://github.com/LSimol/Dynamic-Model-for-Ionospheric-Irregularities.git
cd Dynamic-Model-for-Ionospheric-Irregularities
pip install -r requirements.txt

2. **Data Availability**

The model relies on **pre-processed Swarm A data grids** (binned by season, solar activity, and clock angle). These files are hosted on Zenodo to keep the repository light.

**Download the data automatically**: python download_data.py

This script will fetch the dataset (DOI: 10.5281/zenodo.18327152) and extract it into the data/processed/ directory.

3. **Model Initialisation**

Run the **main processing script** to compute the Spherical Harmonic coefficients for all variables ($N_e$, RODI, $\gamma$(2)). This step generates the .pkl model files used for inference. This run may take several minutes. Enjoy a coffe !

**Run model**:python main_model.py

Output: models/alm_dict_Ne_Swarm_A.pkl, models/alm_dict_RODI_Swarm_A.pkl, etc.

4. **Visualization**

To generate the climatological maps, run the example script. It is fully interactive and will prompt you for the desired geophysical conditions.

**Run**: python example.py

**Interactive Prompts**:

Solar Activity: Enter an F10.7 index (e.g., 120).

Clock Angle: Enter the IMF Clock Angle in degrees (e.g., 90 for Eastward IMF).

Season: Choose Summer, Winter, or Equinox.

Resolution: Choose the maximum Spherical Harmonic degree $L$ (e.g., 30 for smoother maps, or press Enter for full resolution).

Output: The script saves a high-resolution figure ''multi_parameter_view.png'' showing the reconstructed maps for both the Northern and Southern Hemispheres.

### Repository Structure

- ├── data/                   # Data storage (populated by download_data.py)
- ├── models/                 # Trained Model Coefficients (.pkl)
- ├── src/                    # Source Code
- │   ├── spha_model.py       # Core logic: Spherical Harmonics & Interpolation
- │   ├── visualization.py    # Polar plotting routines (Healpix/Matplotlib)
- │   └── config.py           # Physical constants & Grid settings
- ├── download_data.py        # Data fetcher
- ├── main_model.py           # Training script
- ├── example.py              # Inference & Plotting script
- └── requirements.txt        # Dependencies

 ## Contributions and License
Status: This repository is in development phase (v1.0 release). Feel free to contribute by submitting issues, feature requests, or pull requests.
This software is released under the MIT License. If you use this code for your research, please cite the paper referenced above and the software DOI: 10.5281/zenodo.18327152
