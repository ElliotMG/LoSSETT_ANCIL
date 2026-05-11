#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime as dt

# local imports
from lossett_ancil.preprocess.preprocess_kscale import load_kscale_0p5deg, \
    parse_period_id, parse_dri_mod_id, parse_nest_mod_id
from lossett.filtering.integral_filter import filter_field

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    varname = sys.argv[1]
    if "*" in varname:
        varname1, varname2 = varname.split("*")
        print(varname1, varname2)
        varname = varname1+"_times_"+varname2
        product = True
    else:
        product = False
    period = parse_period_id(sys.argv[2])
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,sys.argv[3])
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,sys.argv[4])
    tsteps_per_day = 8
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    # day & hour of simulation
    year = int(sys.argv[5])
    month = int(sys.argv[6])
    day = int(sys.argv[7])
    hour = int(sys.argv[8])
    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"

    # calculation specification
    chunk_latlon = False
    subset_lat = False
    max_r_deg = float(sys.argv[9])
    tsteps = 8
    tchunks = 8
    pchunks = 1
    prec = 1e-10

    # output directory
    OUT_DIR_ROOT = sys.argv[10]
    OUT_DIR = os.path.join(OUT_DIR_ROOT, period, "n640_regrid")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # optional arguments
    try:
        tstep = int(sys.argv[11])
    except (ValueError, IndexError):
        tstep = None
        single_t = False
    else:
        single_t = True
        tchunks = 1

    try:
        plev = int(sys.argv[12])
    except (ValueError, IndexError):
        plev = None
        single_p = False
    else:
        single_p = True
        pchunks = 1

    print(
        "\n\nInput data specifications:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_id}\n"\
        f"nested_model \t= {nest_mod_id}\n"\
        f"datetime \t= {dt_str}\n"\
    )
    print(
        "\nCalculation specifications:\n"\
        f"out_dir \t= {OUT_DIR}\n"\
        f"single_t \t= {single_t}\n"\
        f"single_p \t= {single_p}\n"\
        f"max_r_deg \t= {max_r_deg:.1f}\n"\
        f"tchunks \t= {tchunks}\n"\
        f"pchunks \t= {pchunks}\n"\
        f"subset_lat \t= {subset_lat}\n"\
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
    DATA_DIR = \
        "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"\
        "DYAMOND_SUMMER/n1280_regrid/embedded/n640_regrid/"
    fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_embedded_{dt_str}_n640.nc")
    print(f"\nLoading from {fpath}")
    ds = xr.open_dataset(fpath,mask_and_scale=True,drop_variables="leadtime")
    if product:
        field1 = ds[varname1]
        field2 = ds[varname2]
        field = (field1*field2).rename(varname)
        field.attrs = field1.attrs
        field.attrs["units"] = field1.attrs["units"] + " * " + field2.attrs["units"]
    else:
        field = ds[varname]

    t_str = ""
    if single_t:
        # subset single time
        field = field.isel(time=tstep)
        field = field.expand_dims(dim="time")
        t_str=f"_tstep{tstep}"
    else:
        # subset time; chunk time
        if tsteps_per_day != tsteps:
            t_str = f"_tstep0-{tsteps-1}"
        field = field.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})

    # subset single pressure level
    if single_p:
        field = field.sel(pressure=plev,method="nearest")
        field = field.expand_dims(dim="pressure")
        p_str = f"_p{plev:04d}"
    else:
        field = field.chunk(chunks={"pressure":pchunks})
        p_str = ""

    # subset lat & lon
    subset_str=""
    if subset_lat:
        latmin = -50
        latmax = 50
        field = field.sel(latitude=slice(latmin,latmax))
        subset_str = "_50S-50N"

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_id} filtered {varname} for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_str} (driven by {dri_mod_str}) filtered {varname} for {dt_str}")

    print("\nInput data:\n",field)

    # specify length scales (10 length scales per decade unless 2dx > spacing between consecutive \ell)
    length_scales = np.array(
        [32.,64.,100.,125.,160.,200.,250.,320.,400.,500.,640.,800.,1000.,1250.,1600.,2000.,2500.]
    )
    length_scales *= 1000.0 # convert to m

    # filter velocity component
    field_filtered = filter_field(
        field, control_dict,
        length_scales=length_scales, name=varname+"_filtered"
    )
    print("\n\n\nFiltered field:\n",field_filtered)

    # save to NetCDF
    n_l = len(field_filtered.length_scale)
    L_min = field_filtered.length_scale[0].values/1000
    L_max = field_filtered.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_str}.{dri_mod_str}_0p5deg_{varname}_filtered_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}{subset_str}{p_str}{t_str}.nc"
    )
    print(f"\nFiltered {varname}:\n",field_filtered)
    print(f"\nSaving {field_filtered.name} to NetCDF at location {fpath}.")
    field_filtered.to_netcdf(fpath)

    print("\n\n\nEND.\n")
