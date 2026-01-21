import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview

def plot_polar_hemisphere(map_data, hemisphere='North', vmin=None, vmax=None, 
                          unit_label='', title='', sub=None, cmap='jet'):
    """
    Plots a polar view of the Healpix map following the specific formatting
    requirements (ticks, labels, limits).
    """
    
    # 1. Setup projection parameters based on hemisphere
    if hemisphere == 'North':
        ylim = (np.radians(90), np.radians(50))
        # Ticks for 80, 70, 60
        yticks = np.radians([80, 70, 60])
        yticklabels = ['80°', '', '60°']
        text_radius = np.radians(46)
        # Standard Polar view settings
        proj_args = {
            'flip': 'geo', 
            'phi_convention': 'clockwise',
            'projection_type': 'polar'
        }
    else: # South
        ylim = (np.radians(-90), np.radians(-50))
        # Ticks for -80, -70, -60
        yticks = np.radians([-80, -70, -60])
        yticklabels = ['-80°', '', '-60°']
        text_radius = np.radians(-46)
        # South often requires rotation or specific flip handling depending on convention
        # Using exact settings from snippet for South:
        proj_args = {
            'flip': 'geo', 
            'phi_convention': 'clockwise',
            'projection_type': 'polar'
        }

    # 2. Plot using projview
    # If data is log-scale, handle vmin/vmax outside or pass them directly
    
    projview(
        map_data, 
        coord=['G'], 
        sub=sub, 
        hold=False, 
        graticule=True, 
        cmap=cmap, 
        cbar=True, 
        min=vmin, 
        max=vmax, 
        graticule_labels=True, 
        unit=unit_label, 
        fontsize={'cbar_label': 20, 'cbar_tick_label': 20},
        override_plot_properties={
            'figure_width': 18, 
            'figure_height': 8, 
            'figure_size_ratio': 0.9,
            'cbar_pad': 0.1, 
            'cbar_shrink': 0.7
        },
        **proj_args
    )

    # 3. Apply Custom Formatting (Ticks, Limits, Text)
    plt.ylim(*ylim)
    plt.yticks(yticks, yticklabels, fontsize=20)
    plt.xticks(np.radians([0, 90, 180, 270]), ['', '', '', ''], fontsize=20)

    # 4. Add MLT Labels (00:00, 06:00, 12:00, 18:00)
    # Note: Rotation logic depends on hemisphere/projection, assuming snippet logic
    plt.text(np.radians(0), text_radius, '00:00', ha='center', va='center', fontsize=20)
    plt.text(np.radians(90), text_radius, '06:00', ha='center', va='center', rotation=270, fontsize=20)
    plt.text(np.radians(180), text_radius, '12:00', ha='center', va='center', fontsize=20)
    plt.text(np.radians(270), text_radius, '18:00', ha='center', va='center', rotation=90, fontsize=20)

    # 5. Orientation
    plt.gca().set_theta_zero_location('S')
    
    if title:
        plt.title(title, fontsize=22, y=1.08)
