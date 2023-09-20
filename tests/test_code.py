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


def ij(i, j, k=0):
    return i * 100 + j + k * 1_000_000


def create_zarr(
    vars=["2t", "msl", "10u", "10v"], start=2021, end=2021, frequency=6, k=0
):
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
            data[i, j] = ij(i, j, k)

    root.data = data
    root.dates = dates
    root.attrs["frequency"] = frequency
    root.attrs["resolution"] = 0

    return root


def test_code():
    ds = open_dataset(create_zarr())
    print(len(ds))
    print(ds.dates)

    for i, e in enumerate(ds):
        print(i, e)
        if i > 10:
            break


def test_concat():
    ds = open_dataset(
        create_zarr(start=2021, end=2021),
        create_zarr(start=2022, end=2022),
    )
    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4
    for i, row in enumerate(ds):
        n = i if i < 365 * 4 else i - 365 * 4
        expect = [ij(n, 0), ij(n, 1), ij(n, 2), ij(n, 3)]
        assert (row == expect).all()


def test_join():
    ds = open_dataset(
        create_zarr(vars=["a", "b", "c", "d"]),
        create_zarr(vars=["e", "f", "g", "h"]),
    )

    assert isinstance(ds, Join)
    assert len(ds) == 365 * 4
    for i, row in enumerate(ds):
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


def test_subset_1():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, frequency=12)
    assert len(ds) == 365 * 3 * 2


def test_subset_2():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, start=2022, end=2022)
    assert len(ds) == 365 * 24


def test_subset_3():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, start=2022, end=2022, frequency=12)
    assert len(ds) == 365 * 2


if __name__ == "__main__":
    test_join()
