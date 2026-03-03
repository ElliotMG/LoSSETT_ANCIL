# Script to do some statistics on the LoSSETT output data. This one produces some "ridge plots" for each level in the atmosphere. Thinking of doing one for differing length scales as well.

# READ IN PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
import xarray as xr
import matplotlib as mpl
import cartopy as cpy
import datetime as dt

# READ IN DATA
base_dir = '/gws/nopw/j04/kscale/LoSSETT_data/'
szn = 'DYAMOND_Summer' # Season
dm = 'global_n1280_GAL9' # Driving model
nmr = 'channel_n2560' # Nested model resolution and domain
res = '0p5deg' # Resolution
diri = f'{base_dir}/{szn}/{dm}' # Directory in
var_str = f'n1280GAL9_{res}_inter_scale_transfer_of_kinetic_energy_Lmin_00055_Lmax_01705' # Long file names

# Example for all of DYAMOND Summer
dates = pd.date_range(start="2016-08-01", end="2016-09-09").strftime('%Y%m%d') 

file_pathsG = [f"{diri}/{nmr}_GAL9/{res}/{nmr}_GAL9.{var_str}_{date}.nc" for date in dates]
dsG = xr.open_mfdataset(file_pathsG, combine='by_coords',engine='netcdf4',decode_timedelta=True)

file_pathsR = [f"{diri}/{nmr}_RAL3p2/{res}/{nmr}_RAL3p2.{var_str}_{date}.nc" for date in dates]
dsR = xr.open_mfdataset(file_pathsR, combine='by_coords',engine='netcdf4',decode_timedelta=True)

# PLOTTING
ell=220000
simid = 'channel_n2560_RAL3p2'

da = dsR['Dl_u'].sel(length_scale=ell, method='nearest')
da_flat = da.stack(sample=("latitude", "longitude"))
da_flat = da_flat.drop_vars(['latitude', 'longitude'])
df = da_flat.to_dataframe().reset_index()

pal = sns.color_palette("plasma", len(df['pressure'].unique()))
fig, axes = plt.subplots(len(df['pressure'].unique()), 1, figsize=(5, 5), sharex=True)

pressure_levels = sorted(df['pressure'].unique())

mean_values = []
subplot_positions = []

for i, pressure in enumerate(pressure_levels):
    ax = axes[i]

    subset = df[df['pressure'] == pressure] # Get data for this pressure level
  
    color = pal[i] # Get color for this level
    
    # Plot KDE for this pressure level
    sns.kdeplot(data=subset, x="Dl_u", ax=ax,
                fill=True, color=color, alpha=0.7, linewidth=1.5,
                bw_adjust=1.0, clip_on=True)
    
    # Add the pressure label
    ax.text(-0.00011, 0.5, f"{int(pressure)} hPa", color=color, ha="right", va="center", transform=ax.transAxes)

    # Uncomment this to plot little red lines on each histogram showing the location of the mean. 
    # Can easily be modified to do median as well.
    # mean_value = subset["Dl_u"].mean()
    # ax.axvline(x=mean_value, color='tab:red', linewidth=1, label='Mean')

    # Axis formatting        
    ax.set_yticks([])
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    ax.set_ylabel('')
    
    # Remove all spines except bottom on the last subplot
    if i < len(pressure_levels) - 1:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    ax.set_xlim(-5e-4, 5e-4)
    ax.set_xticks([-5e-4,-4e-4,-3e-4,-2e-4,-1e-4,0,1e-4,2e-4,3e-4,4e-4,5e-4])

    ax.axvline(x=0,color='k',linewidth=0.5)

# Set x-label on the bottom axis only
axes[-1].set_xlabel(r"$\mathcal{D}_{\ell}$ (m$^2$ s$^{-3}$)", fontsize=12)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.1)  # Less extreme spacing
plt.suptitle(f'{simid} {szn} | ' + r'$\ell$' + f'={int(ell*2 / 1000)} km', y=0.92)


###########################################
# Loop to run for all scales - this will store one png per length scale for all levels in the atmosphere 
# for your desired output_path.
for ell in ds['length_scale'].values:
    da = ds['Dl_u'].sel(length_scale=ell, method='nearest')
    da_flat = da.stack(sample=("latitude", "longitude"))
    da_flat = da_flat.drop_vars(['latitude', 'longitude'])
    df = da_flat.to_dataframe().reset_index()
    
    pal = sns.color_palette("plasma", len(df['pressure'].unique()))
    fig, axes = plt.subplots(len(df['pressure'].unique()), 1, figsize=(5, 5), sharex=True)
    
    pressure_levels = sorted(df['pressure'].unique())

    mean_values = []
    subplot_positions = []

    for i, pressure in enumerate(pressure_levels):
        ax = axes[i]
    
        subset = df[df['pressure'] == pressure] # Get data for this pressure level
      
        color = pal[i] # Get color for this level
        
        # Plot KDE for this pressure level
        sns.kdeplot(data=subset, x="Dl_u", ax=ax,
                    fill=True, color=color, alpha=0.7, linewidth=1.5,
                    bw_adjust=1.0, clip_on=True)
        
        # Add the pressure label
        ax.text(-0.00011, 0.5, f"{int(pressure)} hPa", color=color, ha="right", va="center", transform=ax.transAxes)

        # Uncomment this to plot little red lines on each histogram showing the location of the mean. 
        # Can easily be modified to do median as well.
        # mean_value = subset["Dl_u"].mean()
        # ax.axvline(x=mean_value, color='red', linewidth=1, label='Mean')

        # Axis formatting        
        ax.set_yticks([])
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        ax.set_ylabel('')
        
        # Remove all spines except bottom on the last subplot
        if i < len(pressure_levels) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('')
            ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        ax.set_xlim(-5e-4, 5e-4)
        ax.set_xticks([-5e-4,-4e-4,-3e-4,-2e-4,-1e-4,0,1e-4,2e-4,3e-4,4e-4,5e-4])

        ax.axvline(x=0,color='k',linewidth=0.5)

    # Set x-label on the bottom axis only
    axes[-1].set_xlabel(r"$\mathcal{D}_{\ell}$ (m$^2$ s$^{-3}$)", fontsize=12)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1)  # Less extreme spacing
    plt.suptitle(f'{simid} DYAMOND Summer | ' + r'$\ell$' + f'={int(ell / 1000)} km', y=0.92)
    ell_km = f"{int(ell / 1000):04d}" # format to 4sf for sorting outputs
    # Save the figure for the current length scale
    output_path = f'/home/users/emg97/emgPlots/Dlu_hist_ridges/LO_ridge_{simid}_DS_l{ell_km}km_wm.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid memory issues