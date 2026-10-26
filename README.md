# The LoSSETT ancillary scripts

This repository contains scripts for preparing data, running LoSSETT, and
visualising results.

## Installation

Install the ancillary package and the LoSSETT implementation in the same
environment:

```bash
python -m pip install -e .
python -m pip install -e /path/to/LoSSETT
```

The Julia implementation is provided by the `LoSSETT` repository's
`elliotmg-add-quadratic-julia-geometry` branch:

```bash
git clone --branch elliotmg-add-quadratic-julia-geometry \
  https://github.com/ElliotMG/LoSSETT.git "$HOME/LoSSETT-julia"
julia --project="$HOME/LoSSETT-julia/Julia" -e 'using Pkg; Pkg.instantiate()'
```

## ERA5 January 2005 comparison

`Tutorial.py` reads separate regular-grid ERA5 u, v, and vertical-velocity
NetCDF files from:

```text
/gws/ssde/j25b/kscale/USERS/dship/ERA5/3hourly
```

The default files are `era5_u_component_of_wind_200501_3h_0p5deg.nc`,
`era5_v_component_of_wind_200501_3h_0p5deg.nc`, and
`era5_vertical_velocity_200502_3h_0p5deg.nc`. If the latter is instead named
with `200501`, the script uses that file when the default is absent. The
inputs must have identical regular 3-hourly coordinates.

The monthly workflow processes all 248 January 2005 timestamps, keeps the
native 0.5-degree grid, and selects latitudes exactly from 50°S through 50°N.
It compares only:

* Cartesian LoSSETT;
* the tangent-quadratic local spherical-cap approximation.

It does not run the unfinished full-spherical method.

Inspect the input schema first:

```bash
python Tutorial.py --inspect-netcdf
```

Run the complete month:

```bash
python Tutorial.py
```

For a one-timestep smoke test:

```bash
python Tutorial.py --single-timestep 2005 1 1
```

Use `--u-netcdf`, `--v-netcdf`, `--w-netcdf`, `--map-pressure-hpa`, and the
coordinate/variable override options when the files do not expose standard CF
metadata. The default pressure is 200 hPa.

Monthly outputs are written below `Tutorial_monthly_200501/`, with one
timestamp-specific input bundle and two Python result fields per timestamp.
`python_timings.csv` records the measured Python core runtimes. NetCDF loading
and plotting are outside those measurements.

## Julia execution

After the Python preparation completes, run the Julia tutorial over every
prepared timestamp bundle:

```bash
export PYTHON="$(command -v python)"
julia --project="$HOME/LoSSETT-julia/Julia" Tutorial.jl Tutorial_monthly_200501
```

Julia calls the package's `kinetic_energy_transfer` function for Cartesian and
tangent-quadratic geometry. Each method is warmed up before its measured call
so first-use compilation is excluded from the reported runtime.

For every timestamp, the workflow writes Cartesian and tangent-quadratic
Python/Julia fields, side-by-side comparison figures, and difference fields.
The timing files distinguish Python from Julia and Cartesian from
tangent-quadratic calculations. The comparison is between package calls; the
tutorial scripts only handle NetCDF selection, bundle creation, plotting, and
timing.
