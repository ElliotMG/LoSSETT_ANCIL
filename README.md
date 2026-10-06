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

`Tutorial.py` prepares one ERA5 snapshot, runs the original Cartesian Python
core and the Python spherical-geometry branch, and saves one global contour
map for each. It also writes a shared input bundle (`Tutorial_input_YYYYMMDD`)
for `Tutorial.jl`, avoiding a second ERA5 download/remap. The Julia script runs
both its Cartesian and spherical cores, saves their maps, then creates two
difference maps (`Python - Julia`) and prints max-absolute and RMS differences.
Thus running both scripts produces six single-map PNGs (for the default date):

* `Tutorial_cartesian_python_20160801.png`
* `Tutorial_cartesian_julia_20160801.png`
* `Tutorial_spherical_python_20160801.png`
* `Tutorial_spherical_julia_20160801.png`
* `Tutorial_cartesian_python_minus_julia_20160801.png`
* `Tutorial_spherical_python_minus_julia_20160801.png`

All four calculations use the same prepared 4-degree global grid, selected
ERA5 pressure levels and u/v inputs (w is zero), 850 hPa map, and 500 km scale.
The two implementation maps use the same Plate Carrée projection, domain, and
fixed contour scale. Difference maps use a symmetric scale based on each
pair's maximum absolute difference. The difference CLI loads both fields
against the same manifest, rejects mismatched shapes/coordinates, and reports
the number of finite difference cells as well as max-absolute and RMS values
in `m^2 s^-3`.

### Python environment and run

Install the ancillary project first, then replace its default LoSSETT install
with the public branch that contains the spherical Python helpers. That branch
uses Numba and NumExpr, which are not declared by the LoSSETT project metadata:

```powershell
python -m pip install -e .
git clone --branch spherical_geometry https://github.com/ElliotMG/LoSSETT.git "$env:TEMP\LoSSETT-spherical"
python -m pip install -e "$env:TEMP\LoSSETT-spherical"
python -m pip install numba numexpr
python -c "from lossett.calc.spherical_geometry import compute_geometry; from lossett.calc.field_increments import compute_du3_angular_integral_global; print('LoSSETT spherical geometry API is available')"
python .\Tutorial.py 2016 08 01
```

The spherical map calls the `spherical_geometry` branch's `compute_geometry`,
`compute_du3_angular_integral_global`, `get_integration_kernels`, and
`integrate_over_scales` implementations directly, with geometry prepared
in-memory for this reduced tutorial grid. Missing branch helpers are an error;
the script does not substitute a Cartesian calculation. It follows that
branch's current conventions: spherical mollifier normalization but `r dr`
transfer integration (not `R sin(r/R) dr`), uniform angular sample weighting,
and horizontal increments only. Since tutorial w is zero, omitting it does not
change the inputs to the increment norm.

### Julia environment and run

Clone the standalone Julia package branch; its `Project.toml` lives in the
`Julia` subdirectory of the checkout:

```powershell
git clone --branch elliotmg-julia-spherical-geometry https://github.com/ElliotMG/LoSSET.git "$env:TEMP\LoSSET-julia"
julia --project="$env:TEMP\LoSSET-julia\Julia" .\Tutorial.jl .\Tutorial_input_20160801
```

Set `PYTHON` to the Python executable in the prepared plotting environment if
the `python` command is not available to Julia. For another date, run
`Tutorial.py` with that date and pass its matching input directory to
`Tutorial.jl`.

The runtime in each implementation title covers the core call and materialized
result, not catalog access, HEALPix remapping, or plotting. Julia performs an
untimed full-case warm-up to exclude compilation; the spherical Python Numba
angular-integration kernel is separately warmed before timing. Comparisons are
not expected to have zero differences: Cartesian implementations use different
discrete radial/angular quadratures; the spherical Python branch's integration
uses `r dr` while Julia uses `R sin(r/R) dr`; and Julia's spherical core does
not wrap across the longitude seam whereas the Python global workflow rolls
fields periodically. Both spherical paths use uniform angular sample
weighting, but their radial samples and radial integration differ.
