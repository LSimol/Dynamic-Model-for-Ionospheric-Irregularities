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
        Collapses (N_bins, N_obs) -> (N_bins,) by taking the mean of valid values.
        """
        n_bins = data_matrix.shape[0]
        filtered_means = np.full(n_bins, np.nan)
        
        # Optimize: If data is all floats, we can avoid some looping, 
        # but robust implementation requires looping or complex masking
        for i in range(n_bins):
            row = data_matrix[i, :]
            valid = row[~np.isnan(row)]
            
            if len(valid) == 0:
                continue
            
            median = np.median(valid)
            mad = sigma * np.median(np.abs(valid - median))
            
            # Keep values within median +/- MAD
            clean = valid[(valid >= median - mad) & (valid <= median + mad)]
            
            if len(clean) > 0:
                filtered_means[i] = np.mean(clean)
                
        return filtered_means

    def fill_gaps_iterative(self, grid_2d):
        """
        Implements the user's iterative neighbor-averaging logic to fill NaNs.
        """
        # Work on a copy to avoid modifying original
        data = grid_2d.copy()
        nrows, ncols = data.shape
        
        # Limit iterations to prevent infinite loops if grid is empty
        max_iter = 100 
        iteration = 0
        
        while np.isnan(np.nansum(data)) and iteration < max_iter:
            data_prev = data.copy()
            for i in range(nrows):
                for j in range(ncols):
                    if np.isnan(data[i, j]):
                        # Neighbor slice
                        r_min, r_max = max(0, i-1), min(nrows, i+2)
                        c_min, c_max = max(0, j-1), min(ncols, j+2)
                        
                        window = data_prev[r_min:r_max, c_min:c_max]
                        
                        # Only fill if there are valid neighbors
                        if not np.all(np.isnan(window)):
                            data[i, j] = np.nanmean(window)
            
            # Safety check: if no change, stop
            if np.allclose(data, data_prev, equal_nan=True):
                break
            iteration += 1
            
        return data

    def grid_to_healpix(self, lat_grid, lon_grid, data_grid):
        """
        Convert a 2D Lat/Lon grid to a Healpix map.
        """
        hp_map = np.full(hp.nside2npix(self.nside), np.nan, dtype=np.double)
        
        # Robust check for dimensions
        rows, cols = data_grid.shape
        if rows != len(lat_grid) or cols != len(lon_grid):
            # Fallback for safety: iterate min dimensions
            rows = min(rows, len(lat_grid))
            cols = min(cols, len(lon_grid))

        for i in range(rows):
            for j in range(cols):
                val = data_grid[i, j]
                
                if not np.isnan(val):
                    theta = np.radians(90 - lat_grid[i])
                    phi = np.radians(lon_grid[j])
                    pix_idx = hp.ang2pix(self.nside, theta, phi)
                    hp_map[pix_idx] = val
                 
        return hp_map

    def compute_coeffs(self, hp_map):
        # Final safety fill for SPHA (Healpix doesn't like NaNs)
        mask = np.isnan(hp_map)
        if np.any(mask):
            hp_map[mask] = np.nanmean(hp_map)
        
        alm = hp.map2alm(hp_map, lmax=self.lmax, mmax=self.mmax)
        cln = hp.anafast(hp_map, lmax=self.lmax)
        return alm, cln

    def reconstruct_map(self, alm, custom_lmax=None):
        """
        Reconstruct map from alm coefficients.
        If custom_lmax is provided, filter out higher-degree coefficients
        while preserving the array size, preventing 'Wrong alm size' errors.
        """
        # If no custom L is set, or it's higher than the model's native resolution
        if custom_lmax is None or int(custom_lmax) >= self.lmax:
            return hp.alm2map(alm, nside=self.nside, lmax=self.lmax, mmax=self.mmax)
        
        # If custom L is lower, we create a "Step Filter"
        # 1.0 for l <= custom_l, 0.0 for l > custom_l
        l_cutoff = int(custom_lmax)
        fl = np.zeros(self.lmax + 1)
        fl[:l_cutoff + 1] = 1.0
        
        # Apply the filter (Multiplies Alm by the window function)
        alm_filtered = hp.almxfl(alm, fl)
        
        # Reconstruct using the ORIGINAL lmax dimensions (so Healpy is happy)
        # but the high-frequency data is now effectively zero.
        return hp.alm2map(alm_filtered, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def interpolate_coefficients(self, alm_dict, sa_input, ca_input, season_input, sa_levels, ca_levels):
        # ... (Identical to previous correct version) ...
        solar_activities = list(sa_levels.values())
        clock_angles = list(ca_levels.values())
        
        if sa_input in solar_activities and ca_input in clock_angles:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            ca_tag = next(k for k, v in ca_levels.items() if v == ca_input)
            return alm_dict[sa_tag][season_input][ca_tag]

        sorted_angles = np.array(clock_angles + [clock_angles[0] + 2 * np.pi])
        sorted_keys = list(ca_levels.keys()) + [list(ca_levels.keys())[0]]

        idx_upper = np.searchsorted(sorted_angles, ca_input)
        if idx_upper == 0: idx_upper = 1
        if idx_upper >= len(sorted_angles): idx_upper = len(sorted_angles) - 1

        ca_lower = sorted_angles[idx_upper - 1]
        ca_upper = sorted_angles[idx_upper]
        ca_lower_tag = sorted_keys[idx_upper - 1]
        ca_upper_tag = sorted_keys[idx_upper]

        if sa_input in solar_activities:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            coef_upper = alm_dict[sa_tag][season_input][ca_upper_tag]
            coef_lower = alm_dict[sa_tag][season_input][ca_lower_tag]
            return interp1d([ca_lower, ca_upper], np.vstack([coef_lower, coef_upper]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)
        else:
            if sa_input > max(solar_activities):
                sa_upper, sa_lower = sorted(solar_activities)[-1], sorted(solar_activities)[-2]
            elif sa_input < min(solar_activities):
                sa_upper, sa_lower = sorted(solar_activities)[1], sorted(solar_activities)[0]
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
            
            return interp1d([sa_lower, sa_upper], np.vstack([interp_sa_lower, interp_sa_upper]), axis=0, kind='linear', fill_value='extrapolate')(sa_input)
