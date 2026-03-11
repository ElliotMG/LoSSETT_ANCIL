import numpy as np
import xarray as xr
import warnings
import os,sys
import intake
import healpy
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic
import datetime as dt

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

def get_nn_lon_lat_index(nside, lons, lats):
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("latitude", lats), ("longitude", lons)],
    )

cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']

sim = cat['ERA5']
zoom=7
ds = sim(zoom=zoom).to_dask() # Zoom=6 is about 1 deg

supersampling = {"longitude": 4, "latitude": 4}

idx = get_nn_lon_lat_index(
    2**zoom, 
    np.linspace(-180, 180, supersampling['longitude']*180), 
    np.linspace(-90, 90, supersampling['latitude']*90)
)

y = int(sys.argv[1])
m = int(sys.argv[2])
d = int(sys.argv[3])

# In this example we're just doing 2D on each layer to save having to convert ERA5 omega to w
u_lon_lat = ds.u.isel(cell=idx).sel(time=slice(dt.datetime(2016,8,1),dt.datetime(2016,8,1))).coarsen(supersampling).mean()
v_lon_lat = ds.v.isel(cell=idx).sel(time=slice(dt.datetime(2016,8,1),dt.datetime(2016,8,1))).coarsen(supersampling).mean()
w_lon_lat = xr.zeros_like(u_lon_lat)
w_lon_lat.name='w'
# omega_lon_lat = ds.w.isel(time=slice(0,1), cell=idx).coarsen(supersampling).mean()

ds_u_3D = xr.merge([u_lon_lat, v_lon_lat, w_lon_lat])
ds_u_3D = ds_u_3D.rename({'level':'pressure'})
plevs = [10,70,100,150,200,250,300,400,500,600,700,850,925]
ds_u_3D = ds_u_3D.sel(pressure=plevs).compute()

control_dict = {
    "max_r": 10.0,
    "max_r_units": "deg",
    "angle_precision": 1e-10,
    "x_coord_name": "longitude",
    "x_coord_units": "deg",
    "x_coord_boundary": "periodic",
    "y_coord_name": "latitude",
    "y_coord_units": "deg",
    "y_coord_boundary": np.nan
}

length_scales = np.array(
    [110,220,330,500,640,800,1000,1250,1600,2000,2500,3200,4000,5000]
)
length_scales = 1000.0 * length_scales # convert to m; ensure float

# calculate kinetic DR indicator
Dl_u = calc_inter_scale_energy_transfer_kinetic(
    ds_u_3D, control_dict,
    length_scales=length_scales
)

# ensure correct dimension ordering
Dl_u = Dl_u.transpose("length_scale","time","pressure","latitude","longitude")

model = 'ERA5'
var = 'inter_scale_transfer_of_kinetic_energy'
L_min = length_scales[0]
L_max = length_scales[-1]
dt_str = f"{datetime.y:04d}{datetime.m:02d}{datetime.d:02d}"

fname = f"{model}_{var}_Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}.nc"

Dl_u.to_netcdf(fname)
print(f'File {fname} saved')