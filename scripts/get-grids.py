#!/usr/bin/env python3

import os
import sys

import climetlab as cml
import numpy as np

grid = sys.argv[1]

ds = cml.load_source("file", f"{grid}.grib")


field = ds[0]

latitudes, longitudes = field.grid_points()

path = f"grid-{grid}.npz"

np.savez(
    "tmp.npz",
    latitudes=latitudes,
    longitudes=longitudes,
)
os.rename("tmp.npz", path)
