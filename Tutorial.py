"""Read regular-grid NetCDF winds and compare three LoSSETT geometry methods."""

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np


DEFAULT_NETCDF_PATH = Path(
    "/gws/ssde/j25b/kscale/DATA/ENSEMBLE/outdir_20160801T0000Z/"
    "rosie_ens_kscale_ctc/engl_em00/profile_200/"
    "20160801_20160801T0000Z_global_profile_3hourly_200_05deg.nc"
)
LENGTH_SCALE_METRES = 500_000.0
MAP_PRESSURE_HPA = 200.0
MAP_LIMIT = 5e-5
ORIGIN_LATITUDE_CHUNK = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "date",
        nargs="*",
        type=int,
        metavar="DATE_COMPONENT",
        help="date to select from the input NetCDF (default: 2016 8 1)",
    )
    parser.add_argument(
        "--input-netcdf",
        type=Path,
        default=DEFAULT_NETCDF_PATH,
        help=f"regular global lat/lon NetCDF input (default: {DEFAULT_NETCDF_PATH})",
    )
    parser.add_argument(
        "--map-pressure-hpa",
        type=float,
        default=MAP_PRESSURE_HPA,
        help=f"pressure level to select and compare (default: {MAP_PRESSURE_HPA} hPa)",
    )
    parser.add_argument(
        "--input-pressure-hpa",
        type=float,
        help="required if the file has no CF pressure coordinate and contains one fixed pressure level",
    )
    parser.add_argument(
        "--inspect-netcdf",
        action="store_true",
        help="print the input NetCDF schema and exit without calculating",
    )
    parser.add_argument("--time-coordinate", help="override CF time-coordinate detection")
    parser.add_argument(
        "--pressure-coordinate", help="override CF pressure-coordinate detection"
    )
    parser.add_argument(
        "--latitude-coordinate", help="override CF latitude-coordinate detection"
    )
    parser.add_argument(
        "--longitude-coordinate", help="override CF longitude-coordinate detection"
    )
    parser.add_argument("--u-variable", help="override eastward-wind variable detection")
    parser.add_argument("--v-variable", help="override northward-wind variable detection")
    parser.add_argument(
        "--plot-result",
        type=Path,
        help="plot one previously calculated field instead of preparing inputs",
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
        choices=("cartesian", "spherical", "tangent_quadratic"),
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
    if args.inspect_netcdf and (
        args.plot_result is not None or args.plot_difference is not None
    ):
        parser.error("--inspect-netcdf cannot be combined with plotting options")
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
    map_pressure_hpa = float(manifest["map_pressure_hpa"])
    if float(manifest["length_scale_m"]) != LENGTH_SCALE_METRES:
        raise ValueError("Difference fields do not use the requested 500 km scale")
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
    pressure_tag = _pressure_tag(map_pressure_hpa)
    output = Path(
        f"Tutorial_{geometry}_python_minus_julia_{date_compact}_{pressure_tag}.png"
    )
    color_limit = max_absolute if max_absolute > 0 else MAP_LIMIT
    plot_map(
        difference,
        latitude,
        longitude,
        f"Python - Julia ({geometry}, {map_pressure_hpa:g} hPa, 500 km), m^2 s^-3; "
        f"max|diff|={max_absolute:.3e}, RMS={rms:.3e}",
        output,
        color_limit,
        colorbar_label="Python - Julia difference (m^2 s^-3)",
    )


def inspect_netcdf_schema(path):
    import xarray as xr

    if not path.is_file():
        raise FileNotFoundError(
            f"Input NetCDF file does not exist: {path}\n"
            "On JASMIN, run this command from a node that can access the /gws path."
        )
    with xr.open_dataset(path) as dataset:
        print(f"NetCDF schema: {path}")
        print(f"Dimensions: {dict(dataset.sizes)}")
        for name, coordinate in dataset.coords.items():
            print(
                f"Coordinate {name!r}: dims={coordinate.dims}, "
                f"shape={coordinate.shape}, attrs={coordinate.attrs}, "
                f"encoding={coordinate.encoding}"
            )
        for name, variable in dataset.data_vars.items():
            print(
                f"Variable {name!r}: dims={variable.dims}, "
                f"shape={variable.shape}, attrs={variable.attrs}"
            )


def _metadata_matches(coordinate, role):
    attrs = dict(coordinate.encoding)
    attrs.update(coordinate.attrs)
    standard_name = str(attrs.get("standard_name", "")).lower()
    axis = str(attrs.get("axis", "")).upper()
    units = str(attrs.get("units", "")).lower().replace(" ", "")
    if role == "time":
        return standard_name == "time" or axis == "T" or "since" in units
    if role == "latitude":
        return (
            standard_name == "latitude"
            or axis == "Y"
            or units in ("degrees_north", "degree_north", "degreesnorth")
        )
    if role == "longitude":
        return (
            standard_name == "longitude"
            or axis == "X"
            or units in ("degrees_east", "degree_east", "degreeseast")
        )
    if role == "pressure":
        return (
            standard_name == "air_pressure"
            or axis == "Z" and units in ("pa", "hpa", "mbar", "millibar")
        )
    raise ValueError(f"Unknown coordinate role: {role}")


def _resolve_coordinate(dataset, override, role):
    if override is not None:
        if override not in dataset.variables:
            raise ValueError(
                f"Requested {role} coordinate {override!r} is not a NetCDF "
                f"variable/coordinate. Available variables: {list(dataset.variables)}"
            )
        return override
    candidates = [
        name
        for name, coordinate in dataset.coords.items()
        if _metadata_matches(coordinate, role)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify exactly one {role} coordinate from CF metadata "
            f"(found {candidates}). Use --inspect-netcdf and pass the "
            f"appropriate --{role}-coordinate NAME option."
        )
    return candidates[0]


def _resolve_wind_variable(dataset, override, standard_name):
    if override is not None:
        if override not in dataset.data_vars:
            raise ValueError(
                f"Requested wind variable {override!r} is not a data variable. "
                f"Available variables: {list(dataset.data_vars)}"
            )
        return override
    candidates = [
        name
        for name, variable in dataset.data_vars.items()
        if str(variable.attrs.get("standard_name", "")).lower() == standard_name
    ]
    if len(candidates) != 1:
        axis = "u-variable" if standard_name == "eastward_wind" else "v-variable"
        raise ValueError(
            f"Could not identify exactly one {standard_name} variable from CF "
            f"metadata (found {candidates}). Use --inspect-netcdf and pass "
            f"--{axis} NAME."
        )
    return candidates[0]


def _pressure_in_hpa(coordinate):
    units = str(coordinate.attrs.get("units", "")).strip().lower()
    if units in ("pa", "pascal", "pascals"):
        return np.asarray(coordinate.values, dtype=np.float64) / 100.0
    if units in ("hpa", "mbar", "millibar", "millibars"):
        return np.asarray(coordinate.values, dtype=np.float64)
    raise ValueError(
        f"Pressure coordinate has unsupported or missing units {units!r}. "
        "Expected Pa, hPa, mbar, or millibar; inspect the file and correct "
        "its metadata before running the tutorial."
    )


def _same_day_time_index(coordinate, requested_date):
    try:
        values = np.asarray(coordinate.values).astype("datetime64[ns]")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Time coordinate {coordinate.name!r} is not decoded as standard "
            "datetime values. Inspect the NetCDF time metadata and encoding."
        ) from error
    day_start = np.datetime64(requested_date, "D")
    same_day = np.flatnonzero(values.astype("datetime64[D]") == day_start)
    if same_day.size == 0:
        raise ValueError(
            f"No samples on {requested_date} in time coordinate "
            f"{coordinate.name!r}."
        )
    day_start_ns = day_start.astype("datetime64[ns]")
    return int(same_day[np.argmin(np.abs(values[same_day] - day_start_ns))])


def _pressure_tag(pressure_hpa):
    return f"{pressure_hpa:g}hPa"


def _fixed_profile_pressure(path, explicit_pressure_hpa):
    if explicit_pressure_hpa is not None:
        return explicit_pressure_hpa, "explicit --input-pressure-hpa option"
    directory_name = path.parent.name
    if directory_name.startswith("profile_"):
        suffix = directory_name.removeprefix("profile_")
        try:
            return float(suffix), f"input directory {directory_name!r}"
        except ValueError:
            pass
    raise ValueError(
        "The file has no identifiable pressure coordinate. Supply "
        "--input-pressure-hpa explicitly (or use a profile_NNN directory "
        "whose name unambiguously gives the fixed level)."
    )


def _selected_wind_field(
    dataset,
    variable_name,
    time_dimension,
    pressure_dimension,
    latitude_dimension,
    longitude_dimension,
    time_index,
    pressure_index,
    fixed_pressure_hpa,
):
    variable = dataset[variable_name]
    if time_dimension not in variable.dims:
        raise ValueError(
            f"Wind variable {variable_name!r} has no time dimension "
            f"{time_dimension!r}; dimensions are {variable.dims}."
        )
    if (
        pressure_dimension is not None
        and pressure_dimension not in variable.dims
        and dataset.sizes[pressure_dimension] != 1
    ):
        raise ValueError(
            f"Wind variable {variable_name!r} has no pressure dimension "
            f"{pressure_dimension!r}, but the coordinate has multiple levels."
        )
    for dimension in (latitude_dimension, longitude_dimension):
        if dimension not in variable.dims:
            raise ValueError(
                f"Wind variable {variable_name!r} has no dimension "
                f"{dimension!r}; dimensions are {variable.dims}."
            )
    selections = {time_dimension: [time_index]}
    if pressure_dimension is not None and pressure_dimension in variable.dims:
        selections[pressure_dimension] = [pressure_index]
    variable = variable.isel(selections)
    extra_dims = [
        dimension
        for dimension in variable.dims
        if dimension
        not in (
            time_dimension,
            pressure_dimension,
            latitude_dimension,
            longitude_dimension,
        )
    ]
    for dimension in extra_dims:
        if variable.sizes[dimension] != 1:
            raise ValueError(
                f"Wind variable {variable_name!r} has unsupported non-singleton "
                f"dimension {dimension!r} of size {variable.sizes[dimension]}. "
                "Select a single member/realization before running."
            )
        variable = variable.isel({dimension: 0}, drop=True)
    if "pressure" in variable.coords and "pressure" not in variable.dims:
        variable = variable.drop_vars("pressure")
    variable = variable.reset_coords(drop=True)
    rename_dims = {
        time_dimension: "time",
        latitude_dimension: "latitude",
        longitude_dimension: "longitude",
    }
    if pressure_dimension is not None and pressure_dimension in variable.dims:
        rename_dims[pressure_dimension] = "pressure"
    variable = variable.rename(
        {old: new for old, new in rename_dims.items() if old != new}
    )
    if "pressure" not in variable.dims:
        variable = variable.expand_dims(pressure=[fixed_pressure_hpa])
    return variable.transpose("time", "pressure", "latitude", "longitude")


def prepare_netcdf_inputs(path, requested_date, args):
    import xarray as xr

    if not path.is_file():
        raise FileNotFoundError(
            f"Input NetCDF file does not exist: {path}\n"
            "On JASMIN, run this command from a node that can access the /gws path, "
            "or pass --input-netcdf with a readable local copy."
        )

    with xr.open_dataset(path) as source:
        try:
            time_name = _resolve_coordinate(
                source, args.time_coordinate, "time"
            )
            latitude_name = _resolve_coordinate(
                source, args.latitude_coordinate, "latitude"
            )
            longitude_name = _resolve_coordinate(
                source, args.longitude_coordinate, "longitude"
            )
            if args.pressure_coordinate is not None:
                pressure_name = _resolve_coordinate(
                    source, args.pressure_coordinate, "pressure"
                )
            else:
                pressure_candidates = [
                    name
                    for name, coordinate in source.coords.items()
                    if _metadata_matches(coordinate, "pressure")
                ]
                if len(pressure_candidates) > 1:
                    raise ValueError(
                        "Found multiple CF pressure coordinates "
                        f"{pressure_candidates}; specify --pressure-coordinate."
                    )
                pressure_name = (
                    pressure_candidates[0] if pressure_candidates else None
                )
            u_name = _resolve_wind_variable(
                source, args.u_variable, "eastward_wind"
            )
            v_name = _resolve_wind_variable(
                source, args.v_variable, "northward_wind"
            )
        except ValueError as error:
            raise ValueError(
                f"{error}\nInput schema for {path}:\n"
                "Run `python Tutorial.py --inspect-netcdf --input-netcdf "
                f"{path}` and use its coordinate/variable names with the "
                "matching --time-coordinate, --pressure-coordinate, "
                "--latitude-coordinate, --longitude-coordinate, --u-variable, "
                "and --v-variable options."
            ) from error

        coordinates = {
            "time": source[time_name],
            "latitude": source[latitude_name],
            "longitude": source[longitude_name],
        }
        if pressure_name is not None:
            coordinates["pressure"] = source[pressure_name]
        for role, coordinate in coordinates.items():
            if role == "pressure" and coordinate.ndim == 0:
                continue
            if coordinate.ndim != 1:
                raise ValueError(
                    f"{role.capitalize()} coordinate {coordinate.name!r} must "
                    f"be one-dimensional; found dimensions {coordinate.dims}."
                )
        time_dimension = source[time_name].dims[0]
        latitude_dimension = source[latitude_name].dims[0]
        longitude_dimension = source[longitude_name].dims[0]
        time_index = _same_day_time_index(source[time_name], requested_date)
        if pressure_name is not None:
            pressure_hpa = _pressure_in_hpa(source[pressure_name])
            pressure_hpa = np.atleast_1d(pressure_hpa)
            matches = np.flatnonzero(
                np.isclose(
                    pressure_hpa, args.map_pressure_hpa, atol=0.01, rtol=0
                )
            )
            if matches.size != 1:
                raise ValueError(
                    f"Expected exactly one {args.map_pressure_hpa:g} hPa sample "
                    f"in pressure coordinate {pressure_name!r}; found "
                    f"{pressure_hpa.tolist()} hPa."
                )
            pressure_index = int(matches[0])
            fixed_pressure_hpa = args.map_pressure_hpa
            pressure_dimension = (
                source[pressure_name].dims[0]
                if source[pressure_name].ndim == 1
                else None
            )
        else:
            pressure_dimension = None
            fixed_pressure_hpa, pressure_source = _fixed_profile_pressure(
                path, args.input_pressure_hpa
            )
            if not np.isclose(
                fixed_pressure_hpa, args.map_pressure_hpa, atol=0.01, rtol=0
            ):
                raise ValueError(
                    f"Fixed-level input indicates {fixed_pressure_hpa:g} hPa from "
                    f"{pressure_source}, but the requested map level is "
                    f"{args.map_pressure_hpa:g} hPa."
                )
            print(
                f"No pressure coordinate found; treating this as a fixed "
                f"{fixed_pressure_hpa:g} hPa profile based on {pressure_source}."
            )
            pressure_index = None
        latitude = np.asarray(source[latitude_name].values, dtype=np.float64)
        longitude = np.asarray(source[longitude_name].values, dtype=np.float64)
        selected_time = source[time_name].values[time_index]
        u = _selected_wind_field(
            source,
            u_name,
            time_dimension,
            pressure_dimension,
            latitude_dimension,
            longitude_dimension,
            time_index,
            pressure_index,
            fixed_pressure_hpa,
        ).load()
        v = _selected_wind_field(
            source,
            v_name,
            time_dimension,
            pressure_dimension,
            latitude_dimension,
            longitude_dimension,
            time_index,
            pressure_index,
            fixed_pressure_hpa,
        ).load()

    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("Latitude and longitude coordinates must be finite.")
    u = u.assign_coords(
        latitude=latitude,
        longitude=((longitude + 180.0) % 360.0) - 180.0,
        pressure=[args.map_pressure_hpa],
        time=[selected_time],
    ).sortby("latitude").sortby("longitude")
    v = v.assign_coords(
        latitude=latitude,
        longitude=((longitude + 180.0) % 360.0) - 180.0,
        pressure=[args.map_pressure_hpa],
        time=[selected_time],
    ).sortby("latitude").sortby("longitude")

    latitude = np.asarray(u.latitude.values, dtype=np.float64)
    longitude = np.asarray(u.longitude.values, dtype=np.float64)
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("The input needs at least two latitude and longitude points.")
    dlat = np.diff(latitude)
    dlon = np.diff(longitude)
    if not np.allclose(dlat, dlat[0]) or not np.allclose(dlon, dlon[0]):
        raise ValueError("Input latitude/longitude coordinates must form a regular grid.")
    if np.any(dlat <= 0) or np.any(dlon <= 0):
        raise ValueError("Latitude/longitude coordinates must be strictly increasing.")
    if not np.isclose(longitude.size * dlon[0], 360.0, atol=1e-5):
        raise ValueError(
            "Longitude coordinate must cover one global cycle without duplicating "
            "the endpoint."
        )
    if not np.isclose(
        latitude[-1] - latitude[0] + dlat[0], 180.0, atol=max(dlat[0], 1e-5)
    ):
        raise ValueError("Latitude coordinate must cover the global pole-to-pole grid.")
    if np.any(np.abs(latitude) >= 90.0):
        raise ValueError("Exact pole coordinates are unsupported by quadratic geometry.")

    velocities = xr.Dataset(
        {
            "u": u,
            "v": v,
            "w": xr.zeros_like(u).rename("w"),
        }
    )
    velocities = velocities.transpose("time", "pressure", "latitude", "longitude")
    return velocities, selected_time


def write_input_bundle(dataset, output_directory, date_string, selected_time):
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
        f"selected_time={selected_time}",
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
        from lossett.calc.field_increments import compute_du3_angular_integral_subset
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
            "'elliotmg-fix-spherical-kernel-unpacking' branch. Install that branch in this Python "
            "environment as described in README.md; no Cartesian fallback is used."
        ) from error

    return (
        compute_du3_angular_integral_subset,
        RADIUS_EARTH,
        build_distance_bins,
        compute_geometry,
        get_integration_kernels,
        integrate_over_scales,
    )


def require_tangent_quadratic_python_api():
    try:
        from lossett.calc.field_increments import compute_du3_angular_integral_subset
        from lossett.calc.spherical_geometry import (
            RADIUS_EARTH,
            build_distance_bins,
            compute_geometry,
        )
        from lossett.filtering.get_integration_kernels import get_integration_kernels
        from lossett.filtering.integration import integrate_over_scales
    except ImportError as error:
        raise RuntimeError(
            "The Python tangent-quadratic tutorial requires LoSSETT's "
            "'elliotmg-fix-spherical-kernel-unpacking' branch. Install that "
            "branch as described in README.md; no spherical fallback is used."
        ) from error

    return (
        compute_du3_angular_integral_subset,
        RADIUS_EARTH,
        build_distance_bins,
        compute_geometry,
        get_integration_kernels,
        integrate_over_scales,
    )


def calculate_python_spherical(dataset, helpers):
    return _calculate_python_spherical_method(dataset, helpers, "spherical")


def calculate_python_tangent_quadratic(dataset, helpers):
    return _calculate_python_spherical_method(
        dataset, helpers, "tangent_quadratic"
    )


def _calculate_python_spherical_method(dataset, helpers, method):
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
    if np.any(np.abs(latitude) >= 90.0):
        raise ValueError(
            f"Python {method} geometry does not support exact polar grid points"
        )
    dlon = np.diff(longitude)
    dlat = np.diff(latitude)
    if not np.allclose(dlon, dlon[0]) or not np.allclose(dlat, dlat[0]):
        raise ValueError(
            f"Python {method} geometry requires a regular lat/lon grid"
        )
    if longitude[-1] - longitude[0] >= 360.0:
        raise ValueError("Longitude grid must not duplicate the 360-degree endpoint")

    selected_u = np.asarray(dataset.u.isel(time=0).values, dtype=np.float64)
    selected_v = np.asarray(dataset.v.isel(time=0).values, dtype=np.float64)
    pressure = np.asarray(dataset.pressure.values, dtype=np.float64)
    npressure, nlat, nlon = selected_u.shape
    bin_count = nlon // 2
    distance_edges, radii = build_distance_bins(
        bin_count, max_r=np.nextafter(np.pi * earth_radius, np.inf)
    )
    included_radial_bins = np.flatnonzero(radii <= 2.0 * LENGTH_SCALE_METRES)
    if included_radial_bins.size == 0:
        raise ValueError(f"No radial bins fall within the {method} kernel support")
    last_radial_bin = int(included_radial_bins[-1])
    radii_used = radii[: last_radial_bin + 1]
    cap_radius = distance_edges[last_radial_bin + 1]
    delta_longitudes = (np.arange(nlon) - nlon // 2) * dlon[0]
    kernels = get_integration_kernels(
        radii_used,
        [LENGTH_SCALE_METRES],
        normalization="spherical",
        sphere_radius=earth_radius,
        return_deriv=True,
    )
    output = np.empty((nlat, nlon), dtype=np.float64)
    for latitude_start in range(0, nlat, ORIGIN_LATITUDE_CHUNK):
        latitude_stop = min(latitude_start + ORIGIN_LATITUDE_CHUNK, nlat)
        origin_latitudes = latitude[latitude_start:latitude_stop]
        raw_geometry = compute_geometry(
            origin_latitudes,
            latitude,
            delta_longitudes,
            distance_edges,
            radius=earth_radius,
            dtype=np.float64,
            bin_dtype=np.uint8 if bin_count < 256 else np.uint16,
            trig_fns=False,
        )
        initial_bearing = raw_geometry.initial_bearing.values
        final_bearing = raw_geometry.final_bearing.values
        geometry = xr.Dataset(
            {
                "great_circle_distance": raw_geometry.great_circle_distance,
                "great_circle_distance_bin": raw_geometry.great_circle_distance_bin,
                "sine_initial_bearing": (
                    raw_geometry.initial_bearing.dims,
                    np.sin(initial_bearing),
                ),
                "cosine_initial_bearing": (
                    raw_geometry.initial_bearing.dims,
                    np.cos(initial_bearing),
                ),
                "sine_final_bearing": (
                    raw_geometry.final_bearing.dims,
                    np.sin(final_bearing),
                ),
                "cosine_final_bearing": (
                    raw_geometry.final_bearing.dims,
                    np.cos(final_bearing),
                ),
            },
            coords=raw_geometry.coords,
            attrs=raw_geometry.attrs,
        )
        del initial_bearing, final_bearing, raw_geometry
        geometry.attrs["sphere_radius_m"] = earth_radius
        distance_values = geometry.great_circle_distance.values
        bin_values = geometry.great_circle_distance_bin.values
        active_indices = [
            np.where(
                (distance_values[index] < cap_radius)
                & (bin_values[index] <= last_radial_bin)
            )
            for index in range(latitude_stop - latitude_start)
        ]
        print(
            f"LoSSETT.py {method}: origin latitudes "
            f"{latitude_start + 1}-{latitude_stop} of {nlat}"
        )
        angular_field = np.empty(
            (last_radial_bin + 1, latitude_stop - latitude_start, nlon),
            dtype=np.float64,
        )
        for origin_longitude_index in range(nlon):
            longitude_shift = origin_longitude_index - nlon // 2
            for pressure_index in range(npressure):
                rolled_u = xr.DataArray(
                    np.roll(
                        selected_u[pressure_index],
                        -longitude_shift,
                        axis=1,
                    ),
                    dims=("latitude", "longitude"),
                    coords={"latitude": latitude, "longitude": delta_longitudes},
                )
                rolled_v = xr.DataArray(
                    np.roll(
                        selected_v[pressure_index],
                        -longitude_shift,
                        axis=1,
                    ),
                    dims=("latitude", "longitude"),
                    coords={"latitude": latitude, "longitude": delta_longitudes},
                )
                origin_coords = {
                    "origin_latitude": origin_latitudes,
                    "latitude": ("origin_latitude", origin_latitudes),
                }
                origin_u = xr.DataArray(
                    selected_u[
                        pressure_index,
                        latitude_start:latitude_stop,
                        origin_longitude_index,
                    ],
                    dims=("origin_latitude",),
                    coords=origin_coords,
                )
                origin_v = xr.DataArray(
                    selected_v[
                        pressure_index,
                        latitude_start:latitude_stop,
                        origin_longitude_index,
                    ],
                    dims=("origin_latitude",),
                    coords=origin_coords,
                )
                integral = angular_integral(
                    rolled_u,
                    rolled_v,
                    origin_u,
                    origin_v,
                    geometry,
                    active_indices,
                    bin_count,
                    dtype=np.float64,
                    use_angular_weights=False,
                    method=method,
                )
                if isinstance(integral, tuple):
                    integral = integral[0]
                angular_field[:, :, origin_longitude_index] = integral[
                    :, : last_radial_bin + 1
                ].T

        integrand = xr.DataArray(
            angular_field[:, np.newaxis, :, :],
            dims=("r", "pressure", "latitude", "longitude"),
            coords={
                "r": radii_used,
                "pressure": pressure,
                "latitude": origin_latitudes,
                "longitude": longitude,
            },
        )
        transfer = (
            integrate_over_scales(
                integrand,
                kernels.dG_dr * kernels.dG_dr.r,
                ratio_rmax_to_ell=2.0,
            )
            / 4.0
        )
        field = transfer.sel(
            length_scale=LENGTH_SCALE_METRES,
            pressure=MAP_PRESSURE_HPA,
            drop=True,
        )
        output[latitude_start:latitude_stop] = np.asarray(field.compute().values)
    return output


def save_result(field, path):
    np.savetxt(path, field, delimiter=",", fmt="%.17g")
    return path


def save_python_result(field, elapsed, geometry, output_directory, date_string, dataset):
    pressure_tag = _pressure_tag(MAP_PRESSURE_HPA)
    result_path = save_result(
        field, output_directory / f"{geometry}_python_{pressure_tag}.csv"
    )
    output = Path(
        f"Tutorial_{geometry}_python_{date_string.replace('-', '')}_{pressure_tag}.png"
    )
    geometry_labels = {
        "cartesian": "Cartesian",
        "spherical": "Spherical",
        "tangent_quadratic": "Tangent quadratic",
    }
    plot_map(
        field,
        dataset.latitude.values,
        dataset.longitude.values,
        f"LoSSETT.py {geometry_labels[geometry]} "
        f"({MAP_PRESSURE_HPA:g} hPa, 500 km)",
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


def prepare_and_run(args):
    helpers = require_spherical_python_api()
    year, month, day = args.date
    selected_date = dt.datetime(year, month, day)
    date_string = selected_date.strftime("%Y-%m-%d")
    velocities, selected_time = prepare_netcdf_inputs(
        args.input_netcdf, selected_date, args
    )
    if velocities.sizes["time"] != 1:
        raise ValueError(
            f"Expected one selected timestamp for {date_string}, "
            f"got {velocities.sizes['time']}"
        )

    pressure_tag = _pressure_tag(MAP_PRESSURE_HPA)
    output_directory = Path(
        f"Tutorial_input_{selected_date:%Y%m%d}_{pressure_tag}"
    )
    write_input_bundle(velocities, output_directory, date_string, selected_time)
    print(
        f"Prepared {MAP_PRESSURE_HPA:g} hPa shared Python/Julia inputs in "
        f"{output_directory}; selected source time {selected_time}."
    )

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
    tangent_quadratic_helpers = require_tangent_quadratic_python_api()
    started = time.perf_counter()
    tangent_quadratic_field = calculate_python_tangent_quadratic(
        velocities, tangent_quadratic_helpers
    )
    tangent_quadratic_runtime = time.perf_counter() - started
    save_python_result(
        tangent_quadratic_field,
        tangent_quadratic_runtime,
        "tangent_quadratic",
        output_directory,
        date_string,
        velocities,
    )
    print(
        "LoSSETT.py Tangent quadratic core runtime: "
        f"{tangent_quadratic_runtime:.3f} s"
    )
    print(
        "Python results are ready. Run Tutorial.jl with this input directory "
        "to create all three Julia maps and Python - Julia difference maps."
    )


def main():
    global MAP_PRESSURE_HPA
    args = parse_args()
    if args.inspect_netcdf:
        inspect_netcdf_schema(args.input_netcdf)
        return
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
    MAP_PRESSURE_HPA = args.map_pressure_hpa
    prepare_and_run(args)


if __name__ == "__main__":
    main()
