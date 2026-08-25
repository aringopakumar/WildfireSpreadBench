"""
Build dataset_index.json for an HDF5 WildfireSpreadTS dataset.

Run once after converting TIF -> HDF5. Training can then skip opening every
HDF5 on Google Drive during dataset setup.

Example (Colab):
    python src/preprocess/BuildHDF5Index.py \\
        --data_dir /content/drive/MyDrive/newHDF5Data
"""

import argparse
import glob
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tqdm import tqdm

from dataloader.hdf5_cache import read_n_timesteps, save_hdf5_index, load_hdf5_index

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    args = parser.parse_args()

    index = load_hdf5_index(args.data_dir) or {}

    for year in args.years:
        files = sorted(glob.glob(f"{args.data_dir}/{year}/*.hdf5"))
        print(f"{year}: {len(files)} files")
        index.setdefault(str(year), {})

        for h5_path in tqdm(files, desc=str(year)):
            fire_name = os.path.splitext(os.path.basename(h5_path))[0]
            if fire_name in index[str(year)]:
                continue
            n_timesteps = read_n_timesteps(h5_path)
            index[str(year)][fire_name] = {"n_timesteps": n_timesteps}
            save_hdf5_index(args.data_dir, index)

    save_hdf5_index(args.data_dir, index)
    print(f"Wrote {args.data_dir}/dataset_index.json")


if __name__ == "__main__":
    main()
