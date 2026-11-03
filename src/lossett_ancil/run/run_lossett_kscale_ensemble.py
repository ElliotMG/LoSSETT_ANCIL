#!/usr/bin/env python3
"""Run LoSSETT kinetic transfer calculations for K-scale ensemble members."""

import argparse
import logging
from pathlib import Path

import numpy as np

from lossett.calc.calc_inter_scale_transfers import (
    calc_inter_scale_energy_transfer_kinetic,
)
from lossett_ancil.preprocess.preprocess_kscale import (
    _ensemble_members,
    preprocess_ensemble,
)


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
    parser.add_argument("--member", dest="members", action="append")
    parser.add_argument("--members", dest="members", nargs="+")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _output_path(output_dir, ensemble, member, level, max_scale):
    scale = f"{max_scale:g}".replace(".", "p")
    return output_dir / (
        f"{ensemble}_{member}_profile_{level}_ensemble_inter_scale_transfer_"
        f"Lmax_{scale}.nc"
    )


def main(argv=None):
    args = _parser().parse_args(argv)
    input_root = args.input_root_option or args.input_root
    if input_root is None:
        _parser().error("an input root is required")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    members = args.members or _ensemble_members(
        input_root, ensemble=args.ensemble
    )
    if not members:
        logging.error("No ensemble members found under %s", input_root)
        return 1

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
    failures = []
    for member in members:
        output_path = _output_path(
            args.output_dir, args.ensemble, member, args.level, args.max_scale
        )
        if output_path.exists() and not args.overwrite:
            logging.info("Skipping existing output for %s: %s", member, output_path)
            continue
        try:
            ds = preprocess_ensemble(
                input_root,
                member,
                level=args.level,
                ensemble=args.ensemble,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            result = calc_inter_scale_energy_transfer_kinetic(ds, control_dict)
            result.to_netcdf(output_path)
            logging.info("Wrote %s", output_path)
        except Exception:
            logging.exception("Failed to process member %s", member)
            failures.append(member)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
