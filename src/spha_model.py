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
        Input:  (N_bins, N_observations) array. e.g., (64800, 146)
        Output: (N_bins,) array of filtered means.
        """
        # data_matrix rows are spatial bins, columns are raw observations
        n_bins = data_matrix.shape[0]
        filtered_means = np.full(n_bins, np.nan)
        
        # Iterate over each bin (row)
        for i in range(n_bins):
            row_vals = data_matrix[i, :]
            
            # Remove Padding (NaNs)
            valid_vals = row_vals[~np.isnan(row_vals)]
            
            if len(valid_vals) == 0:
                continue
            
            # MAD Statistics
            median = np.median(valid_vals)
            mad = sigma * np.median(np.abs(valid_vals - median))
            
            # Filter Outliers
            mask = (valid_vals >= median - mad) & (valid_vals <= median + mad)
            clean_vals = valid_vals[mask]
            
            if len(clean_vals) > 0:
                filtered_means[i] = np.mean(clean_vals)
                
        return filtered_means

    def grid_to_healpix(self, lat_grid, lon_grid, data_grid):
        """
        Convert a 2D Lat/Lon grid to a Healpix map.
        Expects data_grid to be shape (len(lat_grid), len(lon_grid)).
        """
        hp_map = np.full(hp.nside2npix(self.nside), np.nan, dtype=np.double)
        
        # Iterate over the 2D grid
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
        mask = np.isnan(hp_map)
        if np.any(mask):
            hp_map[mask] = np.nanmean(hp_map)
        
        alm = hp.map2alm(hp_map, lmax=self.lmax, mmax=self.mmax)
        cln = hp.anafast(hp_map, lmax=self.lmax)
        return alm, cln

    def reconstruct_map(self, alm):
        return hp.alm2map(alm, nside=self.nside, lmax=self.lmax, mmax=self.mmax)

    def interpolate_coefficients(self, alm_dict, sa_input, ca_input, season_input,
                                 sa_levels, ca_levels):
        solar_activities = list(sa_levels.values())
        clock_angles = list(ca_levels.values())
        
        # 1. Direct Match
        if sa_input in solar_activities and ca_input in clock_angles:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            ca_tag = next(k for k, v in ca_levels.items() if v == ca_input)
            return alm_dict[sa_tag][season_input][ca_tag]

        # Circular Interpolation Setup
        sorted_angles = np.array(clock_angles + [clock_angles[0] + 2 * np.pi])
        sorted_keys = list(ca_levels.keys()) + [list(ca_levels.keys())[0]]

        idx_upper = np.searchsorted(sorted_angles, ca_input)
        if idx_upper == 0: idx_upper = 1
        if idx_upper >= len(sorted_angles): idx_upper = len(sorted_angles) - 1

        ca_lower = sorted_angles[idx_upper - 1]
        ca_upper = sorted_angles[idx_upper]
        ca_lower_tag = sorted_keys[idx_upper - 1]
        ca_upper_tag = sorted_keys[idx_upper]

        # 2. Circular Interpolation
        if sa_input in solar_activities:
            sa_tag = next(k for k, v in sa_levels.items() if v == sa_input)
            coef_upper = alm_dict[sa_tag][season_input][ca_upper_tag]
            coef_lower = alm_dict[sa_tag][season_input][ca_lower_tag]
            return interp1d([ca_lower, ca_upper], np.vstack([coef_lower, coef_upper]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)

        # 3. Full Interpolation
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
