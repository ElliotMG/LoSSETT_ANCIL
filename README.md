# The LoSSETT ancillaries repo

This repository contains ancillary scripts to aid in the running of LoSSETT (https://github.com/ElliotMG/LoSSETT), preprocessing and visualisation of outputs.

## Repository Structure

Within `src/lossett_ancil`:

* `run` contains Python and Bash scripts for orchestrating the execution of LoSSETT workflows, including on test data.
* `preprocess` contains Python scripts for pre-processing data to match the specific cases in `lossett_ancil/run`, as well as some general processing functions to e.g. perform time interpolation or embed regional models inside driving models.
* `plot` contains various example plotting Python scripts and iPython notebooks.

## Prerequisites
Current distribution of python (Python 3) - built with `xarray` and `numpy`. See `pyproject.toml` for full list of requirements. You must have LoSSETT (https://github.com/ElliotMG/LoSSETT) installed to make use of any of the run scripts.

## Installation

As a user: activate a suitable environment then pip install:

```bash
pip install  git+https://github.com/ElliotMG/LoSSETT_ANCIL.git
```

As a developer: fork then clone the repository (please create a branch before making any changes!), activate a suitable Python environment, navigate to your LoSSETT directory and

```bash
pip install -e .
```

This will install as the user installation but using the editable cloned code. Please commit code improvements and discuss merging with the master branch with Elliot McKinnon-Gray, Dan Shipley, and other users.

## Python and Julia tutorial comparisons

`Tutorial.py` prepares one ERA5 snapshot, runs Python Cartesian,
full-spherical, and `tangent_quadratic` calculations, then writes their fields
as CSV. It also writes a shared input bundle (`Tutorial_input_YYYYMMDD`) for
`Tutorial.jl`, avoiding a second ERA5 download/remap. The Julia script runs the
same three methods, saves their maps, then creates one Python-minus-Julia
difference map per method and prints max-absolute and RMS differences. Running
both scripts produces nine single-map PNGs (for the default date):

* `Tutorial_cartesian_python_20160801.png`
* `Tutorial_cartesian_julia_20160801.png`
* `Tutorial_spherical_python_20160801.png`
* `Tutorial_spherical_julia_20160801.png`
* `Tutorial_cartesian_python_minus_julia_20160801.png`
* `Tutorial_spherical_python_minus_julia_20160801.png`
* `Tutorial_tangent_quadratic_python_20160801.png`
* `Tutorial_tangent_quadratic_julia_20160801.png`
* `Tutorial_tangent_quadratic_python_minus_julia_20160801.png`

All six implementation fields use the same prepared 4-degree global grid,
selected ERA5 pressure levels and u/v inputs (w is zero), 850 hPa map, and
500 km scale. The Python quadratic method calls
`compute_du3_angular_integral_subset(..., method="tangent_quadratic")` and
restricts each origin to a spherical cap containing the radial bins used by
the kernel (bin centers through 2 length scales). It therefore produces a
global field from local quadratic-approximation neighborhoods; it is not the
full-sphere exact spherical calculation. The Julia method uses the same
quadratic initial-bearing correction and a spherical cap bounded at 10 degrees,
but uses a spherical radial Jacobian and retains the vertical increment (w is
zero here). The cap boundaries and radial grids are therefore not identical.
Consequently, the quadratic difference map compares the same geometry
approximation, not numerically identical implementations; radial quadrature
and integration conventions differ. The spherical methods also differ in
radial integration and longitude-seam treatment.

Implementation maps use the same Plate Carrée projection, domain, and fixed
contour scale. Difference maps use a symmetric scale based on each pair's
maximum absolute difference. The difference CLI loads both fields against the
same manifest, rejects mismatched shapes/coordinates, and reports the number
of finite difference cells as well as max-absolute and RMS values in
`m^2 s^-3`. Python writes the implementation fields to
`Tutorial_input_YYYYMMDD/{cartesian,spherical,tangent_quadratic}_python.csv`;
Julia writes the corresponding `*_julia.csv` files in that directory. Each
implementation's runtime is printed and included in the map title.

### Python environment and run

Install the ancillary project first, then replace its default LoSSETT install
with the LoSSETT branch containing the spherical Python helpers and kernel
unpacking fixes. This branch is based on `spherical_geometry` and includes the
subset `tangent_quadratic` API. It uses Numba and NumExpr, which are not
declared by the LoSSETT project metadata:

```bash
python -m pip install -e .
git clone --branch elliotmg-fix-spherical-kernel-unpacking https://github.com/ElliotMG/LoSSETT.git "${TMPDIR:-/tmp}/LoSSETT-spherical"
git -C "${TMPDIR:-/tmp}/LoSSETT-spherical" branch --show-current
python -m pip install -e "${TMPDIR:-/tmp}/LoSSETT-spherical"
python -m pip install numba numexpr
python -c "from lossett.calc.spherical_geometry import compute_geometry; from lossett.calc.field_increments import compute_du3_angular_integral_global, compute_du3_angular_integral_subset; print('LoSSETT spherical and tangent-quadratic APIs are available')"
python Tutorial.py 2016 08 01
```

The full-spherical map calls the `elliotmg-fix-spherical-kernel-unpacking`
branch's `compute_geometry`, `compute_du3_angular_integral_global`,
`get_integration_kernels`, and `integrate_over_scales` implementations
directly. The quadratic map uses `compute_du3_angular_integral_subset` with
per-origin cap-restricted active indices. Geometry is prepared in-memory for
the reduced tutorial grid. Missing branch helpers are an error; the script
does not substitute a Cartesian calculation. The spherical and
tangent-quadratic Python methods use spherical mollifier normalization but
`r dr` transfer integration (not `R sin(r/R) dr`), uniform angular sample
weighting, and horizontal increments only. The separate Cartesian Python
method continues to use the existing Cartesian core.

### Julia environment and run

The Julia quadratic API is on the published `elliotmg-add-quadratic-julia-geometry`
branch at commit `5578c463ab9863aff6b07abdb4ffdaeed6fead41`, based on the
Julia package-precompile fix. The API rejects grids containing either pole;
the tutorial grid's coarsening removes the exact ±90-degree endpoints, and
both calculation paths reject any grid that still contains a pole.
The branch and API have not yet been independently runtime-verified with this
tutorial, so treat this install reference as provisional until that
verification is complete. The standalone Julia package's `Project.toml` lives
in the `Julia` subdirectory:

```bash
git clone --branch elliotmg-add-quadratic-julia-geometry https://github.com/ElliotMG/LoSSET.git "${TMPDIR:-/tmp}/LoSSET-julia"
git -C "${TMPDIR:-/tmp}/LoSSET-julia" checkout 5578c463ab9863aff6b07abdb4ffdaeed6fead41
git -C "${TMPDIR:-/tmp}/LoSSET-julia" rev-parse HEAD
julia --project="${TMPDIR:-/tmp}/LoSSET-julia/Julia" Tutorial.jl Tutorial_input_20160801
```

Confirm that the `rev-parse` output is
`5578c463ab9863aff6b07abdb4ffdaeed6fead41` before running the Julia tutorial.

Set `PYTHON` to the Python executable in the prepared plotting environment if
the `python` command is not available to Julia. For another date, run
`Tutorial.py` with that date and pass its matching input directory to
`Tutorial.jl`.

The runtime in each implementation title covers the core call and materialized
result, not catalog access, HEALPix remapping, or plotting. Julia performs an
untimed full-case warm-up to exclude compilation; the Python Numba
angular-integration kernel is separately warmed before timing. Comparisons are
not expected to have zero differences: Cartesian implementations use different
discrete radial/angular quadratures; Python uses `r dr` while Julia uses the
spherical radial Jacobian; and Julia's full-spherical core does not wrap across
the longitude seam whereas the Python global workflow rolls fields
periodically. The tangent-quadratic methods share the initial-bearing
correction but differ in radial sampling/integration, and Julia also retains
the (zero-valued) vertical component.
