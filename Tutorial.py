"""Compare the Python and Julia LoSSETT kinetic-energy-transfer cores.

Run this script first. It prepares a shared ERA5 input bundle, calculates the
Python result, and saves one contour map. Tutorial.jl consumes that bundle.
"""

import argparse
import datetime as dt
import time
import warnings
from pathlib import Path

import numpy as np


PRESSURE_LEVELS = [10, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925]
LENGTH_SCALE_METRES = 500_000.0
MAP_PRESSURE_HPA = 850
MAP_LIMIT = 5e-5
SUPER_SAMPLING = {"longitude": 4, "latitude": 4}
GRID_COARSENING = {"longitude": 2, "latitude": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "date",
        nargs="*",
        type=int,
        metavar="DATE_COMPONENT",
        help="ERA5 date to process (default: 2016 8 1)",
    )
    parser.add_argument(
        "--plot-result",
        type=Path,
        help="plot a Julia result CSV instead of downloading and calculating ERA5",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="input-bundle manifest required with --plot-result",
    )
    parser.add_argument("--runtime", type=float, help="core runtime in seconds")
    parser.add_argument(
        "--implementation",
        choices=("LoSSETT.py", "LoSSETT.jl"),
        help="implementation label for --plot-result",
    )
    args = parser.parse_args()
    if len(args.date) not in (0, 3):
        parser.error("provide either no date or all three values: YEAR MONTH DAY")
    if not args.date:
        args.date = (2016, 8, 1)
    if args.plot_result is not None and (
        args.manifest is None or args.runtime is None or args.implementation is None
    ):
        parser.error("--plot-result requires --manifest, --runtime, and --implementation")
    return args


def read_manifest(path):
    manifest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        manifest[key] = value
    if manifest.get("format") != "lossett-tutorial-v1":
        raise ValueError(f"Unsupported tutorial input format in {path}")
    return manifest


def parse_vector(manifest, key, dtype=float):
    return np.asarray([dtype(value) for value in manifest[key].split(",")])


def plot_map(field, latitude, longitude, date_string, implementation, runtime):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    contour = ax.contourf(
        longitude,
        latitude,
        field,
        levels=np.linspace(-MAP_LIMIT, MAP_LIMIT, 21),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        extend="both",
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    fig.colorbar(contour, ax=ax, label="Kinetic energy transfer")
    ax.set_title(f"{implementation}: {runtime:.3f} s")

    suffix = "python" if implementation == "LoSSETT.py" else "julia"
    output = Path(f"Tutorial_{suffix}_{date_string.replace('-', '')}.png")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved contour map: {output}")


def plot_result(result_path, manifest_path, implementation, runtime):
    manifest = read_manifest(manifest_path)
    latitude = parse_vector(manifest, "latitude")
    longitude = parse_vector(manifest, "longitude")
    field = np.loadtxt(result_path, delimiter=",")
    expected_shape = (latitude.size, longitude.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"Expected a {expected_shape} contour field in {result_path}, got {field.shape}"
        )
    plot_map(field, latitude, longitude, manifest["date"], implementation, runtime)


def get_nn_lon_lat_index(healpy, xr, nside, longitudes, latitudes):
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    return xr.DataArray(
        healpy.ang2pix(
            nside, longitude_grid, latitude_grid, nest=True, lonlat=True
        ),
        coords=[("latitude", latitudes), ("longitude", longitudes)],
    )


def write_input_bundle(dataset, output_directory, date_string):
    dataset = dataset.transpose("time", "pressure", "latitude", "longitude")
    shape = dataset["u"].shape
    output_directory.mkdir(parents=True, exist_ok=True)

    for name in ("u", "v", "w"):
        values = np.asarray(dataset[name].values, dtype="<f8")
        if values.shape != shape:
            raise ValueError(f"Input variable {name} has unexpected shape {values.shape}")
        values.ravel(order="F").tofile(output_directory / f"{name}.f64le")

    metadata = [
        "format=lossett-tutorial-v1",
        "shape=" + ",".join(str(size) for size in shape),
        "pressure=" + ",".join(f"{value:.17g}" for value in dataset.pressure.values),
        "latitude=" + ",".join(f"{value:.17g}" for value in dataset.latitude.values),
        "longitude=" + ",".join(f"{value:.17g}" for value in dataset.longitude.values),
        f"date={date_string}",
    ]
    (output_directory / "manifest.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )


def run_python_calculation(dataset, date_string):
    from lossett.calc.calc_inter_scale_transfers import (
        calc_inter_scale_energy_transfer_kinetic,
    )

    control = {
        "max_r": 10.0,
        "max_r_units": "deg",
        "angle_precision": 1e-10,
        "x_coord_name": "longitude",
        "x_coord_units": "deg",
        "x_coord_boundary": "periodic",
        "y_coord_name": "latitude",
        "y_coord_units": "deg",
        "y_coord_boundary": np.nan,
    }

    def calculate():
        result = calc_inter_scale_energy_transfer_kinetic(
            dataset,
            control,
            length_scales=np.asarray([LENGTH_SCALE_METRES]),
        )
        return result.compute()

    calculate()
    started = time.perf_counter()
    result = calculate()
    elapsed = time.perf_counter() - started

    result = result.transpose(
        "length_scale", "time", "pressure", "latitude", "longitude"
    )
    field = result.sel(
        length_scale=LENGTH_SCALE_METRES, pressure=MAP_PRESSURE_HPA
    ).isel(time=0)
    plot_map(
        np.asarray(field.values),
        dataset.latitude.values,
        dataset.longitude.values,
        date_string,
        "LoSSETT.py",
        elapsed,
    )
    return elapsed


def prepare_and_run(date):
    import healpy
    import intake
    import xarray as xr

    warnings.filterwarnings("ignore", category=FutureWarning)
    year, month, day = date
    selected_date = dt.datetime(year, month, day)
    date_string = selected_date.strftime("%Y-%m-%d")

    catalog = intake.open_catalog(
        "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
    )["online"]
    dataset = catalog["ERA5"](zoom=7).to_dask()

    index = get_nn_lon_lat_index(
        healpy,
        xr,
        2**7,
        np.linspace(-180, 180, SUPER_SAMPLING["longitude"] * 180),
        np.linspace(-90, 90, SUPER_SAMPLING["latitude"] * 90),
    )

    day_slice = slice(selected_date, selected_date)
    u = dataset.u.isel(cell=index).sel(time=day_slice).coarsen(SUPER_SAMPLING).mean()
    v = dataset.v.isel(cell=index).sel(time=day_slice).coarsen(SUPER_SAMPLING).mean()
    w = xr.zeros_like(u)
    w.name = "w"

    velocities = xr.merge([u, v, w]).rename({"level": "pressure"})
    velocities = velocities.sel(pressure=PRESSURE_LEVELS)
    velocities = velocities.coarsen(
        GRID_COARSENING, boundary="exact"
    ).mean().compute()
    velocities = velocities.transpose("time", "pressure", "latitude", "longitude")
    if velocities.sizes["time"] != 1:
        raise ValueError(
            f"Expected one ERA5 timestamp for {date_string}, got {velocities.sizes['time']}"
        )

    output_directory = Path(f"Tutorial_input_{selected_date:%Y%m%d}")
    write_input_bundle(velocities, output_directory, date_string)
    print(f"Prepared shared Python/Julia inputs in {output_directory}")
    run_python_calculation(velocities, date_string)


def main():
    args = parse_args()
    if args.plot_result is not None:
        plot_result(
            args.plot_result,
            args.manifest,
            args.implementation,
            args.runtime,
        )
        return
    prepare_and_run(args.date)


if __name__ == "__main__":
    main()
