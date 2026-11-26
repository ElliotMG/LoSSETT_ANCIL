#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr

# local imports
from lossett.filtering.integral_filter import filter_field

VARNAMES = ["x_wind", "y_wind"]  # u, v

if __name__ == "__main__":
    # usage: filter_winds_0p5deg.py INPUT_FILE OUT_DIR [max_r_deg]
    in_path = Path(sys.argv[1])
    OUT_DIR = sys.argv[2]
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    length_scales = 1000.0 * np.array([110, 220, 440, 880, 1760, 3410])
    ells = length_scales / 2.
    sep_step = 2 * 55_000  # separation grid step in m
    max_r_deg = (length_scales[-1] + sep_step) / 111_000.0
    if len(sys.argv) > 3:
        max_r_deg = float(sys.argv[3])

    tchunk = 4  # time steps per filter_field call; ~3 GB per step
    prec = 1e-10

    fpath = os.path.join(OUT_DIR, f"{in_path.stem}_uv_filtered.nc")
    if os.path.exists(fpath):
        print(f"{fpath} exists, skipping.")
        sys.exit(0)

    control_dict = {
        "max_r": max_r_deg,
        "max_r_units": "deg",
        "angle_precision": prec,
        "x_coord_name": "longitude",
        "x_coord_units": "deg",
        "x_coord_boundary": "periodic",
        "y_coord_name": "latitude",
        "y_coord_units": "deg",
        "y_coord_boundary": np.nan,
    }

    ds = xr.open_dataset(in_path)
    missing = [v for v in VARNAMES if v not in ds.data_vars]
    if missing:
        sys.exit(f"Missing variables {missing}; found {list(ds.data_vars)}")

    pressure = float(ds["pressure"].values) if "pressure" in ds else None

    print(
        "\n\nCalculation specifications:\n"
        f"input    \t= {in_path}\n"
        f"variables\t= {VARNAMES}\n"
        f"out_dir  \t= {OUT_DIR}\n"
        f"max_r_deg\t= {max_r_deg:.2f}\n"
        f"tchunk   \t= {tchunk}\n"
        f"scales km\t= {length_scales / 1000.}\n"
    )

    out_vars = {}
    for varname in VARNAMES:
        # drop scalar/aux coords (pressure, forecast_*) so they don't
        # interfere with the filter or the concat
        field = (
            ds[varname].reset_coords(drop=True).astype("float32").load()
        )
        nt = field.sizes["time"]
        print(f"\nInput data ({varname}):\n", field)

        pieces = []
        for t0 in range(0, nt, tchunk):
            t1 = min(t0 + tchunk, nt)
            out = filter_field(
                field.isel(time=slice(t0, t1)), control_dict,
                length_scales=ells, name=varname + "_filtered"
            )
            pieces.append(out.astype("float32").load())
            print(f"\n{varname}: filtered time steps {t0}-{t1} of {nt}")
        filtered = xr.concat(pieces, dim="time")

        # check that no length scales were clipped
        if not np.array_equal(filtered.length_scale.values, ells):
            sys.exit(
                f"Unexpected length scales: {filtered.length_scale.values}"
            )
        out_vars[varname + "_filtered"] = filtered

    ds_out = xr.Dataset(out_vars)
    ds_out["length_scale_full"] = ("length_scale", length_scales)
    ds_out["length_scale_full"].attrs["units"] = "m"
    ds_out.attrs.update(
        source_file=str(in_path),
        filter_ells_note="filter half-widths (ell) = length_scale/2, in m",
    )
    if pressure is not None:
        ds_out.attrs["pressure_hPa"] = pressure

    print(f"\nSaving filtered data to {fpath}...")
    tmp = fpath + ".tmp"
    
    for c in ("latitude", "longitude"):
        ds_out[c].attrs.pop("bounds", None)
        
    ds_out.to_netcdf(tmp)
    os.replace(tmp, fpath)  # so a killed job never leaves a half-written file
    print(f"Successfully saved: {fpath}")
    print(f"File size: {os.path.getsize(fpath) / 1e9:.2f} GB")

    ds.close()