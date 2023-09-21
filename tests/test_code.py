# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import zarr

import ecml_tools.data
from ecml_tools.data import (
    Concat,
    Join,
    Select,
    Subset,
    _frequency_to_hours,
    open_dataset,
)


def _(date, var, k=0):
    d = date.year * 10000 + date.month * 100 + date.day
    v = ord(var) - ord("a") + 1
    assert 0 <= k <= 9
    return d * 100 + v + k / 10.0


def create_zarr(
    vars=["a", "b", "c", "d"],
    start=2021,
    end=2021,
    frequency=6,
    resolution="o96",
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
    root.latitudes = np.array([0, 1, 2, 3])
    root.longitudes = np.array([0, 1, 2, 3])
    root.attrs["frequency"] = frequency
    root.attrs["resolution"] = resolution
    root.attrs["name_to_index"] = {k: i for i, k in enumerate(vars)}

    return root


def zarr_from_str(name, mode):
    # test-2021-2021-6h-o96-abcd-0

    args = dict(
        test="test",
        start=2021,
        end=2021,
        frequency=6,
        resolution="o96",
        vars="abcd",
        k=0,
    )

    for name, bit in zip(args, name.split("-")):
        args[name] = bit

    return create_zarr(
        start=int(args["start"]),
        end=int(args["end"]),
        frequency=_frequency_to_hours(args["frequency"]),
        resolution=args["resolution"],
        vars=[x for x in args["vars"]],
        k=int(args["k"]),
    )


zarr.open = zarr_from_str
zarr.convenience.open = zarr_from_str
ecml_tools.data._name_to_path = lambda name: name


def test_concat():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-0",
        "test-2022-2022-6h-o96-abcd-0",
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_join_1():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-0",
        "test-2021-2021-6h-o96-efgh-0",
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


def test_join_2():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-1",
        "test-2021-2021-6h-o96-bdef-2",
    )

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            _(date, "a", 1),
            _(date, "b", 2),
            _(date, "c", 1),
            _(date, "d", 2),
            _(date, "e", 2),
            _(date, "f", 2),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_join_3():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-1",
        "test-2021-2021-6h-o96-abcd-2",
    )

    # TODO: This should trigger a warning about occulted dataset

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            _(date, "a", 2),
            _(date, "b", 2),
            _(date, "c", 2),
            _(date, "d", 2),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_subset_1():
    ds = open_dataset("test-2021-2023-1h-o96-abcd-0", frequency=12)

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
    ds = open_dataset("test-2021-2023-1h-o96-abcd-0", start=2022, end=2022)

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
    ds = open_dataset(
        "test-2021-2023-1h-o96-abcd-0",
        start=2022,
        end=2022,
        frequency=12,
    )

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


def test_select_1():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", select=["b", "d"])

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "b"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_select_2():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", select=["c", "a"])

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "c"), _(date, "a")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_select_3():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", select={"c", "a"})

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "c")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_drop():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", drop="a")

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_reorder_1():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", reorder=["d", "c", "b", "a"])

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "d"), _(date, "c"), _(date, "b"), _(date, "a")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_reorder_2():
    ds = open_dataset("test-2021-2021-6h-o96-abcd-0", reorder=dict(a=3, b=2, c=1, d=0))

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "d"), _(date, "c"), _(date, "b"), _(date, "a")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_constructor_1():
    ds1 = open_dataset("test-2021-2021-6h-o96-abcd-0")

    ds2 = open_dataset("test-2022-2022-6h-o96-abcd-0")

    ds = open_dataset(ds1, ds2)

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_constructor_2():
    ds = open_dataset(
        datasets=["test-2021-2021-6h-o96-abcd-0", "test-2022-2022-6h-o96-abcd-0"]
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_constructor_3():
    ds = open_dataset(
        {"datasets": ["test-2021-2021-6h-o96-abcd-0", "test-2022-2022-6h-o96-abcd-0"]}
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


def test_constructor_4():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-0",
        {"dataset": "test-2022-2022-1h-o96-abcd-0", "frequency": 6},
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()


if __name__ == "__main__":
    test_constructor_4()
