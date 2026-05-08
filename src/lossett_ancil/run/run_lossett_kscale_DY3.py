#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import mo_pack
import numpy as np
import xarray as xr
import datetime as dt
import argparse

from lossett_ancil.preprocess.preprocess_kscale import load_kscale_native, check_longitude, \
    parse_period_id, parse_dri_mod_id, parse_nest_mod_id
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

def get_command_line_arguments():
    parser = argparse.ArgumentParser(
        description = "Compute inter-scale kinetic energy transfers for MO K-Scale DYAMOND3 runs"
    )

    # positional arguments
    parser.add_argument(
        "driving_model",
        type = str,
        help = "specify driving model"
    )
    parser.add_argument(
        "pressure_level",
        type = int,
        help = "specify pressure level (in hPa)"
    )
    parser.add_argument(
        "year",
        type = int,
        help = "specify year"
    )
    parser.add_argument(
        "month",
        type = int,
        help = "specify month"
    )
    parser.add_argument(
        "day",
        type = int,
        help = "specify day"
    )
    parser.add_argument(
        "hour",
        type = int,
        help = "specify hour"
    )
    
    # keyword arguments
    parser.add_argument(
        "-n",
        "--nested_model",
        type = str,
        help = "specify nested model",
        default = "glm"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type = str,
        help = "specify output directory",
        default = "/work/scratch-pw4/dship/LoSSETT/output/kscale/"
    )
    parser.add_argument(
        "-g",
        "--grid",
        type = str,
        help = "specify computation grid",
        default = "native"
    )
    parser.add_argument(
        "-l",
        "--length_scales",
        nargs = "*",
        type = float,
        help = "specify kernel length scales (in km)",
        default = [None]
    )
    parser.add_argument(
        "-t",
        "--tstep",
        type = int,
        help = "specify single time step for computation",
        default = None
    )
    parser.add_argument(
        "--latmin",
        type = float,
        help = "minimum latitude (degrees)",
        default = -90.
    )
    parser.add_argument(
        "--latmax",
        type = float,
        help = "maximum latitude (degrees)",
        default = 90.
    )
    
    args = parser.parse_args()
    
    return args;

ells = {
    "n2560": 1000.0 * np.array(
        [8,16,24,32,40,48,64,80,100,125,160,200,250,320]
    ),
    "n1280": 1000.0 * np.array(
        [16,32,48,64,80,100,125,160,200,250,320,400,500,640,800,1000]
    ),
    "n640": 1000.0 * np.array(
        [32,64,100,125,160,200,250,320,400,500,640,800,1000,1250,1600,2000,2500]
    )
}
deg_to_m = 110000.0 # conversion of latitudinal degrees to m

if __name__ == "__main__":
    # should take all of these from an options file!
    args = get_command_line_arguments()
    # simulation specification
    _period = "DYAMOND3"
    _dri_mod_id = args.driving_model
    _nest_mod_id = args.nested_model
    plev = args.pressure_level
    tsteps_per_file = 4
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    # parse period, driving model, nested model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,_nest_mod_id)

    # day & hour of simulation
    year = args.year
    month = args.month
    day = args.day
    hour = args.hour
    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"

    # calculation specification
    grid = args.grid
    length_scales = np.array(args.length_scales)
    latmin = args.latmin
    latmax = args.latmax
    tsteps = tsteps_per_file
    tchunks = 1
    pchunks = 1
    prec = 1e-10
    tstep = args.tstep
    if tstep is None:
        single_t = False
    else:
        single_t = True

    # output directory
    OUT_DIR_ROOT = args.outdir # required
    OUT_DIR = os.path.join(OUT_DIR_ROOT, period)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # parse native grids
    if nest_mod_id == "glm":
        if "2560" in dri_mod_id:
            native_grid = "n2560"
        elif "1280" in dri_mod_id:
            native_grid = "n1280"
        #endif
    elif "2560" in nest_mod_id:
        native_grid = "n2560"
    elif "1280" in nest_mod_id:
        native_grid = "n1280"
    #endif
        
    # parse calculation grid
    if grid == "native":
        calc_grid = native_grid
        grid_str = ""
    else:
        calc_grid = grid
        grid_str = f"_{calc_grid}"
    #endif
    
    if length_scales[0] == None:
        length_scales = ells[calc_grid]
    else:
        length_scales = 1000.0 * np.array(length_scales)
    #endif

    max_r_deg = 2.0 * (length_scales[-1] / deg_to_m)
    
    print(
        "\n\nInput data specifications:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"nested_model \t= {nest_mod_str} (ID: {nest_mod_id})\n"\
        f"datetime \t= {dt_str}\n"\
    )
    print(
        "\nCalculation specifications:\n"\
        f"out_dir \t= {OUT_DIR}\n"\
        f"single_t \t= {single_t}\n"\
        f"max_r_deg \t= {max_r_deg:.1f}\n"\
        f"tchunks \t= {tchunks}\n"\
        f"pchunks \t= {pchunks}\n"\
        f"lat_min, lat_max \t= {latmin:.4g}, {latmax:.4g}\n"\
        f"calc_grid \t= {calc_grid}\n"\
        f"length_scales \t= {repr(length_scales)}\n"
    )

    control_dict = {
        "max_r": max_r_deg,
        "max_r_units": "deg",
        "angle_precision": prec,
        "x_coord_name": "longitude",
        "x_coord_units": "deg",
        "x_coord_boundary": lon_bound_field,
        "y_coord_name": "latitude",
        "y_coord_units": "deg",
        "y_coord_boundary": lat_bound_field
    }

    # load data
    DATA_DIR = "/work/scratch-pw4/dship/LoSSETT/preprocessed_kscale_data/DYAMOND3/"
    if calc_grid == native_grid:
        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}.nc")
    else:
        fpath = os.path.join(DATA_DIR,f"{calc_grid}_regrid",f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}_{grid}.nc")
    print(f"\nLoading u, v, w from {fpath}")
    ds_u_3D = xr.open_dataset(fpath,mask_and_scale=True,drop_variables="leadtime")
    
    if single_t:
        # subset single time
        ds_u_3D = ds_u_3D.isel(time=tstep)
        ds_u_3D = ds_u_3D.expand_dims(dim="time")
        t_str=f"_tstep{tstep}"
    else:
        t_str = ""
        if tsteps != tsteps_per_file:
            # subset time
            t_str = f"_tstep0-{tsteps-1}"
            ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps))
        # chunk time
        ds_u_3D = ds_u_3D.chunk(chunks={"time":tchunks})

    # subset single pressure level
    ds_u_3D = ds_u_3D.sel(pressure=plev,method="nearest")
    ds_u_3D = ds_u_3D.expand_dims(dim="pressure")
    p_str = f"_p{plev:04d}"
    ds_u_3D.pressure.attrs["units"] = "hPa"

    # subset latitude
    subset_str=""
    ds_u_3D = ds_u_3D.sel(latitude=slice(latmin,latmax))
    # modify to check sign of latmin, latmax to correctly infer South/North
    subset_str = f"_{np.abs(latmin):2.3g}S-{latmax:2.3g}N"

    # check longitude is [-180,180]
    ds_u_3D = check_longitude(ds_u_3D)

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_str} local inter-scale kinetic energy transfers for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_str} (driven by {dri_mod_str}) local inter-scale kinetic energy transfers for {dt_str}")

    print("\nInput data:\n",ds_u_3D)
    
    # calculate kinetic DR indicator
    Dl_u = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict, length_scales=length_scales
    )
    
    # ensure correct dimension ordering
    Dl_u = Dl_u.transpose("length_scale","time","pressure","latitude","longitude")

    # save to NetCDF
    n_l = len(Dl_u.length_scale)
    L_min = Dl_u.length_scale[0].values/1000
    L_max = Dl_u.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_str}.{dri_mod_str}{grid_str}_inter_scale_transfer_of_kinetic_energy_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}{subset_str}{p_str}{t_str}.nc"
    )
    print(f"\n{Dl_u.name}:\n",Dl_u)
    print(f"\nSaving {Dl_u.name} to NetCDF at location {fpath}.")
    Dl_u.to_netcdf(fpath)

    print("\n\n\nEND.\n")
