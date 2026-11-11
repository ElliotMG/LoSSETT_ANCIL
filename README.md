# The LoSSETT ancillaries repo

This repository contains ancillary scripts to aid in the running of LoSSETT (https://github.com/ElliotMG/LoSSETT), preprocessing and visualisation of outputs.

## Repository Structure

Within `src/lossett_ancil`:

* `run` contains Python and Bash scripts for orchestrating the execution of LoSSETT workflows, including on test data. `filter_ensemble_0p5deg.py` filters existing ensemble-mean NetCDF files in place of the input directory, writing compressed `_filtered.nc` outputs. Its `--max-diameter` option is the full LoSSETT maximum separation; the default is 31.0 degrees (3410 km using LoSSETT's 110 km/degree conversion), and LoSSETT adds one separation-grid increment internally. This yields ell values 55,000, 110,000, 220,000, 440,000, 880,000, and 1,705,000 m. The wrapper uses “maximum separation/diameter” terminology; LoSSETT library internals retain the variable name `r` and configuration key `max_r`. Coordinate values in degrees or radians are normalized to float32 degrees.

To write filtered files to a separate directory, pass `--output-dir`, for example:

```bash
python src/lossett_ancil/run/filter_ensemble_0p5deg.py \
  --output-dir /path/to/filtered \
  /path/to/dDKEdt_ensmean_engl_850hPa_0p5deg.nc
```

Without `--output-dir`, outputs are written beside their input files. Input files are never modified.

LoSSETT constructs the full separation-mask grid before filtering, so peak memory is driven by the global 0.5-degree grid and maximum separation rather than only by the six output scales. The runner reduces avoidable memory by converting coordinates and input data to float32 and avoiding an extra output load/copy, but it does not change LoSSETT internals or process scales independently. For HPC jobs, request at least 128 GB RAM for this global mask construction; use 256 GB if the failed job already had 128 GB or if the input contains many time/level slices.
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
