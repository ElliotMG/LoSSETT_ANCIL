#!/usr/bin/env python3
"""Filter existing 0.5-degree ensemble-mean NetCDF files with LoSSETT."""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.filtering.integral_filter import filter_field


DEFAULT_INPUT_DIR = Path(
    "/gws/ssde/j25b/kscale/USERS/emg/data/Ensemble_Data/DS/DKE/0p5deg"
)
DEFAULT_PATTERN = "dDKEdt_ensmean_*.nc"
LEGACY_LENGTH_SCALES_KM = np.array(
    [110, 220, 330, 440, 550, 660, 770, 880, 990, 1100, 1400, 1750, 2200, 2750, 3500],
    dtype=float,
)
FILTER_LENGTH_SCALES_M = 500.0 * LEGACY_LENGTH_SCALES_KM
COMPRESSION_LEVEL = 4


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Input NetCDF files; omit to discover files in --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory to scan when files are omitted (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Input glob when files are omitted (default: {DEFAULT_PATTERN}).",
    )
    parser.add_argument(
        "--data-variable",
        help="Name of the data variable to filter; required if the file has several candidates.",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="Kernel radius in degrees (default: twice the largest filter scale).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing *_filtered.nc outputs.",
    )
    return parser


def _data_variable(ds, requested):
    if requested is not None:
        if requested not in ds.data_vars:
            raise ValueError(
                f"data variable {requested!r} is not present; available variables: "
                f"{', '.join(ds.data_vars)}"
            )
        return requested

    candidates = [
        name for name, variable in ds.data_vars.items()
        if np.issubdtype(variable.dtype, np.number)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "could not select a unique numeric data variable; use --data-variable "
            f"(candidates: {', '.join(candidates) or '<none>'})"
        )
    return candidates[0]


def _coordinate_name(ds, candidates, axis):
    for name in candidates:
        if name in ds.coords and ds[name].ndim == 1:
            return name
    raise ValueError(
        f"could not identify a one-dimensional {axis} coordinate; "
        f"expected one of {', '.join(candidates)}"
    )


def _output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}")


def filter_file(input_path, data_variable=None, max_radius=None, overwrite=False):
    """Filter one input file and return the output path."""
    input_path = Path(input_path)
    output_path = _output_path(input_path)
    if output_path.exists() and not overwrite:
        logging.info("Skipping existing output %s", output_path)
        return output_path

    with xr.open_dataset(input_path, mask_and_scale=True) as ds:
        variable_name = _data_variable(ds, data_variable)
        field = ds[variable_name]
        longitude = _coordinate_name(ds, ("longitude", "lon", "x"), "longitude")
        latitude = _coordinate_name(ds, ("latitude", "lat", "y"), "latitude")
        radius = max_radius
        if radius is None:
            radius = 2.0 * FILTER_LENGTH_SCALES_M[-1] / 111_000.0
        control_dict = {
            "max_r": radius,
            "max_r_units": "deg",
            "angle_precision": 1e-10,
            "x_coord_name": longitude,
            "x_coord_units": ds[longitude].attrs.get("units", "deg"),
            "x_coord_boundary": "periodic",
            "y_coord_name": latitude,
            "y_coord_units": ds[latitude].attrs.get("units", "deg"),
            "y_coord_boundary": np.nan,
        }
        logging.info(
            "Filtering %s (%s) with radius %.2f degrees",
            input_path,
            variable_name,
            radius,
        )
        filtered = filter_field(
            field,
            control_dict,
            length_scales=FILTER_LENGTH_SCALES_M,
            name=f"{variable_name}_filtered",
        )
        filtered.attrs = {**field.attrs, **filtered.attrs}
        filtered = filtered.astype("float32").load()
        output_ds = filtered.to_dataset()
        output_ds.attrs = ds.attrs.copy()
        output_ds.encoding = ds.encoding.copy()

    output_ds.to_netcdf(
        output_path,
        encoding={
            output_ds[filtered.name].name: {
                "dtype": "float32",
                "zlib": True,
                "complevel": COMPRESSION_LEVEL,
            }
        },
    )
    logging.info("Wrote %s", output_path)
    return output_path


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.files:
        input_files = args.files
    else:
        input_files = sorted(
            path for path in args.input_dir.glob(args.pattern)
            if not path.stem.endswith("_filtered")
        )
    if not input_files:
        _parser().error("no input files found")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for input_path in input_files:
        try:
            filter_file(
                input_path,
                data_variable=args.data_variable,
                max_radius=args.max_radius,
                overwrite=args.overwrite,
            )
        except Exception:
            logging.exception("Failed to process %s", input_path)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
