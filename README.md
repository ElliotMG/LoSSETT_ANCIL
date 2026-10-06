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

## Python and Julia tutorial timings

`Tutorial.py` prepares one ERA5 snapshot, runs the Python kinetic-energy-transfer
calculation, and saves one global contour map. It also writes a small shared
input bundle (`Tutorial_input_YYYYMMDD`) for `Tutorial.jl`. The bundle avoids
downloading and remapping ERA5 a second time. An additional 2x spatial
coarsening keeps the Julia core's direct-grid calculation practical while
retaining the global domain. Both maps use the 850 hPa, 500 km
kinetic-energy-transfer field, the same inputs, Plate Carrée projection,
global domain, and contour scale. The Julia spherical implementation does not
wrap across the longitude seam, so values within its 10-degree radius of the
antimeridian may differ from Python's periodic-longitude calculation.

Run the Python script first in an environment with this repository's Python
dependencies and LoSSETT installed:

```powershell
python Tutorial.py 2016 08 01
```

This produces `Tutorial_python_20160801.png` and the Julia input bundle. To use
the native Julia core, clone its public branch and point Julia at the package's
actual project directory (`Julia` is a subdirectory of the LoSSETT checkout):

```powershell
git clone --branch elliotmg-julia-spherical-geometry https://github.com/ElliotMG/LoSSETT.git "$env:TEMP\LoSSETT-julia"
julia --project="$env:TEMP\LoSSETT-julia\Julia" .\Tutorial.jl .\Tutorial_input_20160801
```

The Julia command writes `Tutorial_julia_20160801.png` beside the scripts. For
another date, run both commands with the same date and pass the corresponding
`Tutorial_input_YYYYMMDD` directory to Julia. The reported elapsed time covers
only each LoSSETT core calculation and its materialization; catalog access,
HEALPix remapping, file handoff, and plotting are excluded. Each implementation
runs one untimed full-case warm-up first so first-call compilation is not part
of the reported time. ERA5 data is read from the remote catalog when the
Python script runs.
