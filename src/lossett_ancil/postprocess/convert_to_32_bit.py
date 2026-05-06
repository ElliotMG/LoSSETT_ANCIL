import os
import sys
import xarray as xr
from pathlib import Path

def file_to_32(infile, outdir, in_place=False):
    print("\n")
    if in_place:
        print("Converting in place!")
        outdir = str(infile.parent)
    elif outdir == str(infile.parent):
        print("Cannot output to same directory")
        return None
    # Load the NetCDF file
    ds = xr.open_dataset(infile)

    print("Original dataset:\n", ds)

    # Convert each variable to single precision
    for var in ds.variables:
        if ds[var].dtype == "float64":
            ds[var] = ds[var].astype("float32")

    ds.compute()

    # Save the modified dataset to a new NetCDF file

    if in_place:
        outfile = Path(f"{outdir}/tmp_{infile.name}")
    else:
        outfile = Path(f"{outdir}/{infile.name}")

    print(f"Converting {infile} to {outfile}")

    ds.to_netcdf(outfile, compute=True)

    ds.close()

    if in_place:
        print(f"Replacing {infile} with {outfile}")
        os.replace(outfile, infile)

        outfile = infile
    return outfile;

def files_to_32(indir, pattern, outdir, in_place=False):
    files = Path(indir).glob(pattern)
    n = 0
    for infile in files:

        outfile = file_to_32(infile, outdir, in_place=in_place)

        ds = xr.open_dataset(outfile)
        print("Converted dataset:\n",ds)

        ds.close()
        n += 1

    return n;

def main(indir, pattern, outdir=None, in_place=False):
    if not os.path.exists(indir):
        print(f"\nError: Directory {indir} does not exist. Exiting.")
        sys.exit(1)
    #endif

    if in_place:
        outdir = None
        print(f"\nConverting files matching {pattern} in {indir} to 32bit in place.")
    else:
        if outdir is None:
            outdir = os.path.join(indir,"32bit")
        os.makedirs(outdir, exist_ok=True)
        print(f"\nConverting files matching {pattern} in {indir} to 32bit at {outdir}")
    #endif
    ncopied = files_to_32(
        indir, 
        pattern, 
        outdir, 
        in_place=in_place
    )
                         
    print(f"\nComplete; {ncopied} files converted.")
    return 0;

if __name__ == "__main__":
    indir = str(sys.argv[1])
    pattern = str(sys.argv[2])
    convert_in_place = False

    main(indir, pattern, in_place=convert_in_place)

    
