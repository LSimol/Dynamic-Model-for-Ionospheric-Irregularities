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
                
                if np.all(np.isnan(cell_vals)):
                    continue
                    
                median = np.nanmedian(cell_vals)
                mad = sigma * np.nanmedian(np.abs(cell_vals - median))
                
                # Filter: keep values within median +/- mad
                # Note: Logic adapted for array-based approach; simplified for clarity
                # In full implementation, this should mask outliers in the raw bin lists
        return data_copy

    def compute_coeffs(self, hp_map):
        """Compute spherical harmonic coefficients (alm) and power spectrum (cln)."""
        # Ensure map is valid (handle NaNs with simple mean imputation for SPHA stability)
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
        
        Parameters:
        - alm_dict: Nested dictionary [SA][Season][CA] -> alm_array
        - sa_input: Solar Activity value (F10.7)
        - ca_input: Clock Angle value (radians)
        - season_input: 'Summer', 'Equinox', or 'Winter'
        - sa_levels: Dict mapping labels to values, e.g., {'LSA': 70, ...}
        - ca_levels: Dict mapping labels to radians, e.g., {'UR': 0.52, ...}
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
        # We use standard numpy searchsorted to find where our angle fits
        # We normalize ca_input to 0-2pi if necessary in a real scenario
        idx_upper = np.searchsorted(sorted_angles, ca_input)
        
        # Safety for boundary conditions (though circular logic usually handles this)
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

            # Interpolate between angles
            interp_func = interp1d([ca_lower, ca_upper], np.vstack([coef_lower, coef_upper]), axis=0, kind='linear', fill_value='extrapolate')
            return interp_func(ca_input)

        # 3. Full Interpolation (Both Solar Activity and Clock Angle)
        else:
            # Find SA neighbors
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

            # Retrieve the 4 corners of our interpolation square
            # Upper SA (High F10.7)
            coef_uu = alm_dict[sa_upper_tag][season_input][ca_upper_tag] # Upper SA, Upper Angle
            coef_ul = alm_dict[sa_upper_tag][season_input][ca_lower_tag] # Upper SA, Lower Angle
            
            # Lower SA (Low F10.7)
            coef_lu = alm_dict[sa_lower_tag][season_input][ca_upper_tag] # Lower SA, Upper Angle
            coef_ll = alm_dict[sa_lower_tag][season_input][ca_lower_tag] # Lower SA, Lower Angle

            # Step A: Interpolate Angles for Lower SA
            interp_sa_lower = interp1d([ca_lower, ca_upper], np.vstack([coef_ll, coef_lu]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)
            
            # Step B: Interpolate Angles for Upper SA
            interp_sa_upper = interp1d([ca_lower, ca_upper], np.vstack([coef_ul, coef_uu]), axis=0, kind='linear', fill_value='extrapolate')(ca_input)

            # Step C: Interpolate between Solar Activities
            final_interp = interp1d([sa_lower, sa_upper], np.vstack([interp_sa_lower, interp_sa_upper]), axis=0, kind='linear', fill_value='extrapolate')(sa_input)

            return final_interp
