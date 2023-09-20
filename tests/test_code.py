# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import zarr

from ecml_tools.data import Concat, Join, open_dataset


def ij(i, j):
    return i * 10000 + j


def create_zarr(vars=["2t", "msl", "10u", "10v"], start=2021, end=2021, frequency=6):
    root = zarr.group()

    dates = []
    date = datetime.datetime(start, 1, 1)
    while date.year <= end:
        dates.append(date)
        date += datetime.timedelta(hours=frequency)

    dates = np.array(dates, dtype="datetime64")

    data = np.zeros((len(dates), len(vars)))
    for i in range(len(dates)):
        for j in range(len(vars)):
            data[i, j] = ij(i, j)

    root.data = data
    root.dates = dates
    root.attrs["frequency"] = frequency
    root.attrs["resolution"] = 0

    return root


def test_code():
    root = create_zarr()
    z = open_dataset(root)
    print(len(z))
    print(z.dates)

    for i, e in enumerate(z):
        print(i, e)
        if i > 10:
            break


def test_concat():
    z = open_dataset(
        create_zarr(start=2021, end=2021),
        create_zarr(start=2022, end=2022),
    )
    assert isinstance(z, Concat)
    assert len(z) == 365 * 2 * 4
    for i, row in enumerate(z):
        n = i if i < 365 * 4 else i - 365 * 4
        expect = [ij(n, 0), ij(n, 1), ij(n, 2), ij(n, 3)]
        assert (row == expect).all()


def test_join():
    z = open_dataset(
        create_zarr(vars=["a", "b", "c", "d"]),
        create_zarr(vars=["e", "f", "g", "h"]),
    )

    assert isinstance(z, Join)
    assert len(z) == 365 * 4
    for i, row in enumerate(z):
        n = i
        expect = [
            ij(n, 0),
            ij(n, 1),
            ij(n, 2),
            ij(n, 3),
            ij(n, 0),
            ij(n, 1),
            ij(n, 2),
            ij(n, 3),
        ]
        assert (row == expect).all()


if __name__ == "__main__":
    test_join()
