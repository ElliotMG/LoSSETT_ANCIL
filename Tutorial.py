"""Prepare ERA5 inputs and compare Cartesian and spherical LoSSETT cores."""

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
        help="plot one previously calculated field instead of preparing ERA5",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="input-bundle manifest required with --plot-result or --plot-difference",
    )
    parser.add_argument("--runtime", type=float, help="core runtime in seconds")
    parser.add_argument("--label", help="plot title label for --plot-result")
    parser.add_argument("--output", type=Path, help="output image for --plot-result")
    parser.add_argument(
        "--plot-difference",
        choices=("cartesian", "spherical"),
        help="plot Python minus Julia fields for this geometry",
    )
    parser.add_argument("--python-result", type=Path)
    parser.add_argument("--julia-result", type=Path)
    args = parser.parse_args()
    if len(args.date) not in (0, 3):
        parser.error("provide either no date or all three values: YEAR MONTH DAY")
    if not args.date:
        args.date = (2016, 8, 1)
    if args.plot_result is not None and (
        args.manifest is None
        or args.runtime is None
        or args.label is None
        or args.output is None
    ):
        parser.error("--plot-result requires --manifest, --runtime, --label, and --output")
    if args.plot_difference is not None and (
        args.manifest is None
        or args.python_result is None
        or args.julia_result is None
    ):
        parser.error(
            "--plot-difference requires --manifest, --python-result, and --julia-result"
        )
    if args.plot_result is not None and args.plot_difference is not None:
        parser.error("choose either --plot-result or --plot-difference")
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


def plot_map(
    field,
    latitude,
    longitude,
    label,
    output,
    color_limit,
    runtime=None,
    colorbar_label="Kinetic energy transfer (m^2 s^-3)",
):
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
        levels=np.linspace(-color_limit, color_limit, 21),
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        extend="both",
    )
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    if runtime is None:
        title = label
    else:
        title = f"{label}: {runtime:.3f} s"
    ax.set_title(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved contour map: {output}")


def load_field(result_path, manifest):
    latitude = parse_vector(manifest, "latitude")
    longitude = parse_vector(manifest, "longitude")
    field = np.loadtxt(result_path, delimiter=",", ndmin=2)
    expected_shape = (latitude.size, longitude.size)
    if field.shape != expected_shape:
        raise ValueError(
            f"Expected a {expected_shape} contour field in {result_path}, got {field.shape}"
        )
    return field, latitude, longitude


def plot_result(result_path, manifest_path, label, output, runtime):
    manifest = read_manifest(manifest_path)
    field, latitude, longitude = load_field(result_path, manifest)
    plot_map(field, latitude, longitude, label, output, MAP_LIMIT, runtime)


def plot_difference(python_path, julia_path, manifest_path, geometry):
    manifest = read_manifest(manifest_path)
    if float(manifest["length_scale_m"]) != LENGTH_SCALE_METRES:
        raise ValueError("Difference fields do not use the requested 500 km scale")
    if float(manifest["map_pressure_hpa"]) != MAP_PRESSURE_HPA:
        raise ValueError("Difference fields do not use the requested 850 hPa level")
    python_field, latitude, longitude = load_field(python_path, manifest)
    julia_field, julia_latitude, julia_longitude = load_field(julia_path, manifest)
    if not np.array_equal(latitude, julia_latitude) or not np.array_equal(
        longitude, julia_longitude
    ):
        raise ValueError("Difference fields do not use identical coordinates")

    difference = python_field - julia_field
    finite = np.isfinite(difference)
    if not finite.any():
        raise ValueError("Difference fields contain no finite values")
    finite_difference = difference[finite]
    max_absolute = float(np.max(np.abs(finite_difference)))
    rms = float(np.sqrt(np.mean(np.square(finite_difference, dtype=np.float64))))
    print(
        f"Python - Julia ({geometry}) differences [m^2 s^-3]: "
        f"max_abs={max_absolute:.6e}, rms={rms:.6e}, "
        f"finite={finite.sum()}/{difference.size}"
    )

    date_compact = manifest["date"].replace("-", "")
    output = Path(f"Tutorial_{geometry}_python_minus_julia_{date_compact}.png")
    color_limit = max_absolute if max_absolute > 0 else MAP_LIMIT
    plot_map(
        difference,
        latitude,
        longitude,
        f"Python - Julia ({geometry}, 850 hPa, 500 km), m^2 s^-3; "
        f"max|diff|={max_absolute:.3e}, RMS={rms:.3e}",
        output,
        color_limit,
        colorbar_label="Python - Julia difference (m^2 s^-3)",
    )


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
        f"length_scale_m={LENGTH_SCALE_METRES:.17g}",
        f"map_pressure_hpa={MAP_PRESSURE_HPA:.17g}",
        "pressure=" + ",".join(f"{value:.17g}" for value in dataset.pressure.values),
        "latitude=" + ",".join(f"{value:.17g}" for value in dataset.latitude.values),
        "longitude=" + ",".join(f"{value:.17g}" for value in dataset.longitude.values),
        f"date={date_string}",
    ]
    (output_directory / "manifest.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )


def run_python_calculation(dataset):
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

    started = time.perf_counter()
    result = calculate()
    elapsed = time.perf_counter() - started

    result = result.transpose(
        "length_scale", "time", "pressure", "latitude", "longitude"
    )
    field = result.sel(
        length_scale=LENGTH_SCALE_METRES, pressure=MAP_PRESSURE_HPA
    ).isel(time=0)
    return np.asarray(field.values), elapsed


def require_spherical_python_api():
    try:
        from lossett.calc.field_increments import compute_du3_angular_integral_global
        from lossett.calc.spherical_geometry import (
            RADIUS_EARTH,
            build_distance_bins,
            compute_geometry,
        )
        from lossett.filtering.get_integration_kernels import get_integration_kernels
        from lossett.filtering.integration import integrate_over_scales
    except ImportError as error:
        raise RuntimeError(
            "The Python spherical tutorial requires LoSSETT's "
            "'spherical_geometry' branch. Install that branch in this Python "
            "environment as described in README.md; no Cartesian fallback is used."
        ) from error

    return (
        compute_du3_angular_integral_global,
        RADIUS_EARTH,
        build_distance_bins,
        compute_geometry,
        get_integration_kernels,
        integrate_over_scales,
    )


def calculate_python_spherical(dataset, helpers):
    import xarray as xr

    (
        angular_integral,
        earth_radius,
        build_distance_bins,
        compute_geometry,
        get_integration_kernels,
        integrate_over_scales,
    ) = helpers
    latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
    longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
    dlon = np.diff(longitude)
    dlat = np.diff(latitude)
    if not np.allclose(dlon, dlon[0]) or not np.allclose(dlat, dlat[0]):
        raise ValueError("Python spherical geometry requires a regular lat/lon grid")
    if longitude[-1] - longitude[0] >= 360.0:
        raise ValueError("Longitude grid must not duplicate the 360-degree endpoint")

    selected_u = np.asarray(dataset.u.isel(time=0).values, dtype=np.float64)
    selected_v = np.asarray(dataset.v.isel(time=0).values, dtype=np.float64)
    pressure = np.asarray(dataset.pressure.values, dtype=np.float64)
    npressure, nlat, nlon = selected_u.shape
    bin_count = nlon // 2
    distance_edges, radii = build_distance_bins(
        # Keep exact antipodes inside the final digitize bin.
        bin_count, max_r=np.nextafter(np.pi * earth_radius, np.inf)
    )
    delta_longitudes = (np.arange(nlon) - nlon // 2) * dlon[0]
    geometry = compute_geometry(
        latitude,
        latitude,
        delta_longitudes,
        distance_edges,
        radius=earth_radius,
        dtype=np.float64,
        bin_dtype=np.uint8 if bin_count < 256 else np.uint16,
        trig_fns=True,
    )

    angular_field = np.empty(
        (bin_count, npressure, nlat, nlon), dtype=np.float64
    )
    for origin_longitude_index in range(nlon):
        longitude_shift = origin_longitude_index - nlon // 2
        for pressure_index in range(npressure):
            rolled_u = xr.DataArray(
                np.roll(selected_u[pressure_index], -longitude_shift, axis=1),
                dims=("latitude", "longitude"),
                coords={"latitude": latitude, "longitude": delta_longitudes},
            )
            rolled_v = xr.DataArray(
                np.roll(selected_v[pressure_index], -longitude_shift, axis=1),
                dims=("latitude", "longitude"),
                coords={"latitude": latitude, "longitude": delta_longitudes},
            )
            origin_u = xr.DataArray(
                selected_u[pressure_index, :, origin_longitude_index],
                dims=("origin_latitude",),
                coords={"origin_latitude": latitude},
            )
            origin_v = xr.DataArray(
                selected_v[pressure_index, :, origin_longitude_index],
                dims=("origin_latitude",),
                coords={"origin_latitude": latitude},
            )
            integral, _ = angular_integral(
                rolled_u,
                rolled_v,
                origin_u,
                origin_v,
                geometry,
                bin_count,
                dtype=np.float64,
                use_angular_weights=False,
            )
            angular_field[:, pressure_index, :, origin_longitude_index] = integral.T

    integrand = xr.DataArray(
        angular_field,
        dims=("r", "pressure", "latitude", "longitude"),
        coords={
            "r": radii,
            "pressure": pressure,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    kernels = get_integration_kernels(
        radii,
        [LENGTH_SCALE_METRES],
        normalization="spherical",
        sphere_radius=earth_radius,
        return_deriv=True,
    )
    transfer = (
        integrate_over_scales(
            integrand,
            kernels.dG_dr * kernels.dG_dr.r,
            ratio_rmax_to_ell=2.0,
        )
        / 4.0
    )
    transfer = transfer.sel(
        length_scale=LENGTH_SCALE_METRES, pressure=MAP_PRESSURE_HPA
    )
    return np.asarray(transfer.compute().values)


def save_result(field, path):
    np.savetxt(path, field, delimiter=",", fmt="%.17g")
    return path


def save_python_result(field, elapsed, geometry, output_directory, date_string, dataset):
    result_path = save_result(
        field, output_directory / f"{geometry}_python.csv"
    )
    output = Path(
        f"Tutorial_{geometry}_python_{date_string.replace('-', '')}.png"
    )
    plot_map(
        field,
        dataset.latitude.values,
        dataset.longitude.values,
        f"LoSSETT.py {geometry.capitalize()} (850 hPa, 500 km)",
        output,
        MAP_LIMIT,
        elapsed,
    )
    return result_path


def warmup_spherical_python_numba():
    from lossett.calc.angular_integration import angular_integral_by_distance_bin

    angular_integral_by_distance_bin(
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.uint8),
        1,
    )


def prepare_and_run(date):
    import healpy
    import intake
    import xarray as xr

    warnings.filterwarnings("ignore", category=FutureWarning)
    helpers = require_spherical_python_api()
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

    cartesian_field, cartesian_runtime = run_python_calculation(
        velocities
    )
    save_python_result(
        cartesian_field,
        cartesian_runtime,
        "cartesian",
        output_directory,
        date_string,
        velocities,
    )
    print(f"LoSSETT.py Cartesian core runtime: {cartesian_runtime:.3f} s")

    warmup_spherical_python_numba()
    started = time.perf_counter()
    spherical_field = calculate_python_spherical(velocities, helpers)
    spherical_runtime = time.perf_counter() - started
    save_python_result(
        spherical_field,
        spherical_runtime,
        "spherical",
        output_directory,
        date_string,
        velocities,
    )
    print(f"LoSSETT.py Spherical core runtime: {spherical_runtime:.3f} s")
    print(
        "Python results are ready. Run Tutorial.jl with this input directory "
        "to create the two Julia maps and both Python - Julia difference maps."
    )


def main():
    args = parse_args()
    if args.plot_result is not None:
        plot_result(
            args.plot_result,
            args.manifest,
            args.label,
            args.output,
            args.runtime,
        )
        return
    if args.plot_difference is not None:
        plot_difference(
            args.python_result,
            args.julia_result,
            args.manifest,
            args.plot_difference,
        )
        return
    prepare_and_run(args.date)


if __name__ == "__main__":
    main()
