#!/usr/bin/env python3
"""Run LoSSETT kinetic transfer calculations for K-scale ensemble members."""

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np

from lossett.calc.calc_inter_scale_transfers import (
    calc_inter_scale_energy_transfer_kinetic,
)
from lossett_ancil.preprocess.preprocess_kscale import preprocess_ensemble


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", nargs="?", type=Path)
    parser.add_argument("--input-root", dest="input_root_option", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-scale",
        "--max-scale-deg",
        dest="max_scale",
        required=True,
        type=float,
    )
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument(
        "--ensemble",
        choices=("engl", "CTC_N2560_GAL9", "CTC_N2560_RAL3p2"),
        default="engl",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument(
        "--day-index",
        type=int,
        help="Zero-based day offset from --first-date to process.",
    )
    parser.add_argument(
        "--first-date",
        default="2016-08-01",
        help="First date used with --day-index (default: 2016-08-01).",
    )
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _output_path(
    output_dir,
    member,
    level,
    l_min,
    l_max,
    start_time=None,
    end_time=None,
):
    # LoSSETT reports length scales in metres; filenames use kilometres to
    # match the naming convention of the existing runners.
    date_suffix = ""
    if start_time is not None:
        start_date = start_time[:10].replace("-", "")
        end_date = (end_time or start_time)[:10].replace("-", "")
        date_suffix = f"_{start_date}" if start_date == end_date else (
            f"_{start_date}_{end_date}"
        )
    return output_dir / (
        f"{member}_profile_{level}_ensemble_inter_scale_transfer_"
        f"Lmin_{l_min:05.0f}_Lmax_{l_max:05.0f}{date_suffix}.nc"
    )


def main(argv=None):
    args = _parser().parse_args(argv)
    input_root = args.input_root_option or args.input_root
    if input_root is None:
        _parser().error("an input root is required")
    if args.day_index is not None:
        # Convert the Slurm array index into one calendar day. The runner
        # deliberately handles one member and one day per process.
        if args.day_index < 0:
            _parser().error("--day-index must be non-negative")
        if args.start_time is not None or args.end_time is not None:
            _parser().error(
                "--day-index cannot be combined with --start-time or --end-time"
            )
        try:
            day = dt.date.fromisoformat(args.first_date) + dt.timedelta(
                days=args.day_index
            )
        except ValueError as exc:
            _parser().error(f"--first-date must be YYYY-MM-DD: {exc}")
        args.start_time = day.isoformat()
        args.end_time = day.isoformat()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dict = {
        "max_r": args.max_scale,
        "max_r_units": "deg",
        "angle_precision": 1e-10,
        "x_coord_name": "longitude",
        "x_coord_units": "deg",
        "x_coord_boundary": "periodic",
        "y_coord_name": "latitude",
        "y_coord_units": "deg",
        "y_coord_boundary": np.nan,
    }
    try:
        ds = preprocess_ensemble(
            input_root,
            args.member,
            level=args.level,
            ensemble=args.ensemble,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        # Calculate before naming the file because the actual Lmax depends on
        # the scale grid returned by LoSSETT, not directly on --max-scale-deg.
        result = calc_inter_scale_energy_transfer_kinetic(ds, control_dict)
        l_min = result.length_scale[0].values / 1000
        l_max = result.length_scale[-1].values / 1000
        output_path = _output_path(
            args.output_dir,
            args.member,
            args.level,
            l_min,
            l_max,
            args.start_time,
            args.end_time,
        )
        if output_path.exists() and not args.overwrite:
            logging.info("Skipping existing output for %s", args.member)
            return 0
        result.to_netcdf(output_path)
        logging.info("Wrote %s", output_path)
    except Exception:
        logging.exception("Failed to process member %s", args.member)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
