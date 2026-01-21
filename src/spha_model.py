import numpy as np
import healpy as hp
import copy
from scipy.interpolate import interp1d

class SPHAModel:
    def __init__(self, nside=32):
        self.nside = nside
        self.lmax = 2 * nside
        self.mmax = self.lmax

    def apply_mad_filter(self, data_matrix, sigma=1.4826):
        """
        Apply Median Absolute Deviation (MAD) filtering to a 2D matrix (lat x lon).
        """
        data_copy = copy.deepcopy(data_matrix)
        rows, cols = data_copy.shape[:2]
        
        for i in range(rows):
            for j in range(cols):
                # Handle cell arrays (list of values in a bin)
                cell_vals = data_copy[i][j] if hasattr(data_copy[i][j], '__iter__') else data_copy[i,j]
                
                # Skip if empty or NaN
                if np.all(np.isnan(cell_vals)):
                    continue
                    
                median = np.nanmedian(cell_vals)
                mad = sigma * np.nanmedian(np.abs(cell_vals - median))
                
                # Filter: keep values within median +/- mad
                # Ideally, we return the mean of the filtered values for the map
                valid_indices = np.where((cell_vals >= median - mad) & (cell_vals <= median + mad))
                filtered_vals = cell_vals[valid_indices]
                
                if len(filtered_vals) > 0:
                    data_copy[i, j] = np.mean(filtered_vals)
                else:
                    data_copy[i, j] = np.nan
                    
        return data_copy.astype(float)

    def grid_to_healpix(self, lat_grid, lon_grid, data_grid):
        """
        Convert a Lat/Lon grid to a Healpix map.
        """
        hp_map = np.full(hp.nside2npix(self.nside), np.nan, dtype=np.double)
        
        # Create pixel indices for the grid coordinates
        # lat_grid is 1D array of latitudes, lon_grid is 1D array of longitudes
        # data_grid is 2D array (lat, lon)
        
        # We iterate over the grid and fill the corresponding Healpix pixels
        # Note: This is a simple nearest-neighbor mapping for demonstration.
        # For high-res grids, multiple grid points might map to one pixel (averaging needed)
        
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                val = data_grid[i, j]
                if not np.isnan(val):
                    # Convert Lat/Lon to colatitude/longitude in radians
                    theta = np.radians(90 - lat)
                    phi = np.radians(lon)
                    
                    pix_idx = hp.ang2pix(self.nside, theta, phi)
                    hp_map[pix_idx] = val
                 
        return hp_map

    def compute_coeffs(self, hp_map):
        """Compute spherical harmonic coefficients (alm) and power spectrum (cln)."""
        # Ensure map is valid (handle NaNs with mean imputation)
        mask = np.isnan(hp_map)
        if np.any(mask):
            hp_map[mask] = np.nanmean(hp_map)
        
        alm = hp.map2alm(hp_map, lmax=self.lmax, mmax=self.mmax)
        cln = hp.anafast(hp_map, lmax=self.lmax)
        return alm, cln

    def reconstruct_map(self, alm):
        """Reconstruct map from alm coefficients."""
        return hp.alm2map(alm, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def interpolate_coefficients(self, alm_dict, sa_input, ca_input, season_input,
                                 sa_levels, ca_levels):
        """
        Interpolate Spherical Harmonic coefficients based on Solar Activity (SA)
        and Clock Angle (CA).
        """
        solar_activities = list(sa_levels.values())
        clock_angles = list(ca_levels.values())
        
        # 1. Direct Match
        if sa_input in solar_activities and ca_input in clock_angles:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            ca_tag = next(k for k, v in ca_levels.items() if v == ca_input)
            return alm_dict[sa_tag][season_input][ca_tag]

        # Prepare for circular interpolation (Clock Angle)
        sorted_angles = np.array(clock_angles + [clock_angles[0] + 2 * np.pi])
        sorted_keys = list(ca_levels.keys()) + [list(ca_levels.keys())[0]]

        # Find CA neighbors
        idx_upper = np.searchsorted(sorted_angles, ca_input)
        
        if idx_upper == 0: idx_upper = 1
        if idx_upper >= len(sorted_angles): idx_upper = len(sorted_angles) - 1

        ca_lower = sorted_angles[idx_upper - 1]
        ca_upper = sorted_angles[idx_upper]
        
        ca_lower_tag = sorted_keys[idx_upper - 1]
        ca_upper_tag = sorted_keys[idx_upper]

        # 2. Circular Interpolation ONLY (Direct Solar Activity Match)
        if sa_input in solar_activities:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            
            coef_upper = alm_dict[sa_tag][season_input][ca_upper_tag]
            coef_lower = alm_dict[sa_tag][season_input][ca_lower_tag]

            interp_func = interp1d([ca_lower, ca_upper], np.vstack([coef_lower, coef_upper]), axis=0, kind='linear', fill_value='extrapolate')
            return interp_func(ca_input)

        # 3. Full Interpolation (Both Solar Activity and Clock Angle)
        else:
            if sa_input > max(solar_activities):
                sa_upper = sorted(solar_activities)[-1]
                sa_lower = sorted(solar_activities)[-2]
            elif sa_input < min(solar_activities):
                sa_upper = sorted(solar_activities)[1]
                sa_lower = sorted(solar_activities)[0]
            else:
                sa_upper = min([s for s in solar_activities if s >= sa_input])
                sa_lower = max([s for s in solar_activities if s <= sa_input])

            sa_upper_tag = next(k for k, v in sa_levels.items() if v == sa_upper)
            sa_lower_tag = next(k for k, v in sa_levels.items() if v == sa_lower)

            coef_uu = alm_dict[sa_upper_tag][season_input][ca_upper_tag]
            coef_ul = alm_dict[sa_upper_tag][season_input][ca_lower_tag]
            coef_lu = alm_dict[sa_lower_tag][season_input][ca_upper_tag]
            coef_ll = alm_dict[sa_lower_tag][season_input][ca_lower_tag]

            interp_sa_lower = interp1d([ca_lower, ca_upper], np.vstack([coef_ll, coef_lu]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)
            interp_sa_upper = interp1d([ca_lower, ca_upper], np.vstack([coef_ul, coef_uu]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)

            final_interp = interp1d([sa_lower, sa_upper], np.vstack([interp_sa_lower, interp_sa_upper]), axis=0, kind='linear', fill_value='extrapolate')(sa_input)

            return final_interp
