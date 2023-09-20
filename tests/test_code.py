# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import zarr

from ecml_tools.data import Concat, Join, Subset, open_dataset


def _(date, var, k=0):
    d = date.year * 10000 + date.month * 100 + date.day
    v = ord(var) - ord("a") + 1
    return d * 100 + v + k * 1_000_000_000


def create_zarr(
    vars=["a", "b", "c", "d"],
    start=2021,
    end=2021,
    frequency=6,
    k=0,
):
    root = zarr.group()

    dates = []
    date = datetime.datetime(start, 1, 1)
    while date.year <= end:
        dates.append(date)
        date += datetime.timedelta(hours=frequency)

    dates = np.array(dates, dtype="datetime64")

    data = np.zeros((len(dates), len(vars)))
    for i, date in enumerate(dates):
        for j, var in enumerate(vars):
            data[i, j] = _(date.astype(object), var, k)

    root.data = data
    root.dates = dates
    root.attrs["frequency"] = frequency
    root.attrs["resolution"] = 0

    return root


def test_concat():
    ds = open_dataset(
        create_zarr(start=2021, end=2021),
        create_zarr(start=2022, end=2022),
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for i, row in enumerate(ds):
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_join():
    ds = open_dataset(
        create_zarr(vars=["a", "b", "c", "d"]),
        create_zarr(vars=["e", "f", "g", "h"]),
    )

    assert isinstance(ds, Join)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            _(date, "a"),
            _(date, "b"),
            _(date, "c"),
            _(date, "d"),
            _(date, "e"),
            _(date, "f"),
            _(date, "g"),
            _(date, "h"),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_subset_1():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, frequency=12)

    assert isinstance(ds, Subset)
    assert len(ds) == 365 * 3 * 2

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            _(date, "a"),
            _(date, "b"),
            _(date, "c"),
            _(date, "d"),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=12)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_subset_2():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, start=2022, end=2022)

    assert isinstance(ds, Subset)
    assert len(ds) == 365 * 24

    dates = []
    date = datetime.datetime(2022, 1, 1)

    for row in ds:
        expect = [
            _(date, "a"),
            _(date, "b"),
            _(date, "c"),
            _(date, "d"),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=1)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_subset_3():
    z = create_zarr(start=2021, end=2023, frequency=1)
    ds = open_dataset(z, start=2022, end=2022, frequency=12)

    assert isinstance(ds, Subset)
    assert not isinstance(ds.dataset, Subset)
    assert len(ds) == 365 * 2

    dates = []
    date = datetime.datetime(2022, 1, 1)

    for row in ds:
        expect = [
            _(date, "a"),
            _(date, "b"),
            _(date, "c"),
            _(date, "d"),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=12)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


if __name__ == "__main__":
    test_join()
