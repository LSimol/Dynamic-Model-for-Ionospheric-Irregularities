# Dynamic-Model-for-Ionospheric-Irregularities

This repository provides the **Dynamic Model for Ionospheric Irregularities**, a Python tool designed to characterizes ionospheric irregularity and turbulence indices for different magnetic latitudes, magnetic local times, seasons, solar activity and interplanetary magnetic field orientations at high latitudes of both hemispheres

### Description

This model is based on **spherical harmonic decomposition** and provides statistical maps of :
1) **Electron density**
2) The electron density rate of change as expressed by the **RODI** index
3) A **proxy for ionosphere turbulent processes**
   
The model operates under user-specified conditions for solar activity levels, local seasons, and IMF orientations.

Further details regarding the model and its performance can be found in **Mestici et al., 2025 (submitted)**. Citation details will be updated upon publication.

### Usage

If you plan to use the Model in your research or publications, please make sure to cite the corresponding paper. 

An example on how to get and plot the outputs of the model is provided in the example.py/ipynb file.

### Updates and Contributions

This repository is **currently in the development phase (alpha version)** and will be continuously improved.
Feel free to contribute by submitting issues, feature requests, or pull requests

## Data Availability
The pre-processed data (`.npy` grids) required to run the model are available on Zenodo.

[![DOI](https://zenodo.org/records/18327152/files/data_Ne(6).zip?download=1)

**Usage:**
1. Download the `processed_data.zip` file from the link above.
2. Extract the contents directly into the `data/processed/` folder.
