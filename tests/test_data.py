# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from functools import cache

import numpy as np
import zarr

import ecml_tools.data
from ecml_tools.data import (
    Concat,
    Join,
    Rename,
    Select,
    Subset,
    Ensemble,
    _as_first_date,
    _as_last_date,
    _frequency_to_hours,
    open_dataset,
)

VALUES = 10


@cache
def _(date, var, k=0, e=0):
    d = date.year * 10000 + date.month * 100 + date.day
    v = ord(var) - ord("a") + 1
    assert 0 <= k <= 9
    assert 0 <= e <= 9

    return np.array(
        [d * 100 + v + k / 10.0 + w / 100.0 + e / 1000.0 for w in range(VALUES)]
    )


def create_zarr(
    vars=["a", "b", "c", "d"],
    start=2021,
    end=2021,
    frequency=6,
    resolution="o96",
    k=0,
    ensemble=None,
):
    root = zarr.group()

    dates = []
    date = datetime.datetime(start, 1, 1)
    while date.year <= end:
        dates.append(date)
        date += datetime.timedelta(hours=frequency)

    dates = np.array(dates, dtype="datetime64")

    if ensemble is not None:
        data = np.zeros((len(dates), len(vars), ensemble, VALUES))
        for i, date in enumerate(dates):
            for j, var in enumerate(vars):
                for e in range(ensemble):
                    data[i, j, e] = _(date.astype(object), var, k, e)
    else:
        data = np.zeros((len(dates), len(vars), VALUES))
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

    root.mean = np.mean(data, axis=0)
    root.stdev = np.std(data, axis=0)
    root.maximum = np.max(data, axis=0)
    root.minimum = np.min(data, axis=0)

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
        ensemble=None,
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
        ensemble=int(args["ensemble"]) if args["ensemble"] is not None else None,
    )


zarr.convenience.open = zarr_from_str
ecml_tools.data._name_to_path = lambda name, zarr_root: name


def same_stats(ds1, ds2, vars1, vars2=None):
    if vars2 is None:
        vars2 = vars1

    vars1 = [v for v in vars1]
    vars2 = [v for v in vars2]
    for v1, v2 in zip(vars1, vars2):
        idx1 = ds1.name_to_index[v1]
        idx2 = ds2.name_to_index[v2]
        assert (ds1.statistics["mean"][idx1] == ds2.statistics["mean"][idx2]).all()
        assert (ds1.statistics["stdev"][idx1] == ds2.statistics["stdev"][idx2]).all()
        assert (
            ds1.statistics["maximum"][idx1] == ds2.statistics["maximum"][idx2]
        ).all()
        assert (
            ds1.statistics["minimum"][idx1] == ds2.statistics["minimum"][idx2]
        ).all()


def slices(ds, start=None, end=None, step=None):
    if start is None:
        start = 5
    if end is None:
        end = len(ds) - 5
    if step is None:
        step = len(ds) // 10

    s = ds[start:end:step]

    assert s[0].shape == ds[0].shape, (
        s.shape,
        ds.shape,
        len(list(range(start, end, step))),
        list(range(start, end, step)),
    )

    for i, n in enumerate(range(start, end, step)):
        assert (s[i] == ds[n]).all()


def test_concat():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd",
        "test-2022-2022-6h-o96-abcd",
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4

    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert ds.shape == (365 * 2 * 4, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_join_1():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd",
        "test-2021-2021-6h-o96-efgh",
    )

    assert isinstance(ds, Join)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert ds.name_to_index == {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
    }

    assert ds.shape == (365 * 4, 8, VALUES)

    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_join_2():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-1",
        "test-2021-2021-6h-o96-bdef-2",
    )

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d", "e", "f"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

    assert ds.shape == (365 * 4, 6, VALUES)

    same_stats(
        ds,
        open_dataset(
            "test-2021-2021-6h-o96-ac-1",
            "test-2021-2021-6h-o96-bdef-2",
        ),
        "abcdef",
    )
    slices(ds)


def test_join_3():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd-1",
        "test-2021-2021-6h-o96-abcd-2",
    )

    # TODO: This should trigger a warning about occulted dataset

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd-2"), "abcd")
    slices(ds)


def test_subset_1():
    ds = open_dataset("test-2021-2023-1h-o96-abcd", frequency=12)

    assert isinstance(ds, Subset)
    assert len(ds) == 365 * 3 * 2
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 3 * 2, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_2():
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start=2022, end=2022)

    assert isinstance(ds, Subset)
    assert len(ds) == 365 * 24
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 24, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_3():
    ds = open_dataset(
        "test-2021-2023-1h-o96-abcd",
        start=2022,
        end=2022,
        frequency=12,
    )

    assert isinstance(ds, Subset)
    assert not isinstance(ds.dataset, Subset)
    assert len(ds) == 365 * 2
    assert len([row for row in ds]) == len(ds)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 2, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_4():
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start=202206, end=202208)

    assert isinstance(ds, Subset)
    assert len(ds) == (30 + 31 + 31) * 24
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2022, 6, 1)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == ((30 + 31 + 31) * 24, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_5():
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start=20220601, end=20220831)

    assert isinstance(ds, Subset)
    assert len(ds) == (30 + 31 + 31) * 24
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2022, 6, 1)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == ((30 + 31 + 31) * 24, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_6():
    ds = open_dataset(
        "test-2021-2023-1h-o96-abcd",
        start="2022-06-01",
        end="2022-08-31",
    )

    assert isinstance(ds, Subset)
    assert len(ds) == (30 + 31 + 31) * 24
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2022, 6, 1)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == ((30 + 31 + 31) * 24, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_subset_7():
    ds = open_dataset("test-2021-2023-1h-o96-abcd", start="2022-06", end="2022-08")

    assert isinstance(ds, Subset)
    assert len(ds) == (30 + 31 + 31) * 24
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2022, 6, 1)

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

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == ((30 + 31 + 31) * 24, 4, VALUES)

    same_stats(ds, open_dataset("test-2021-2023-1h-o96-abcd"), "abcd")
    slices(ds)


def test_select_1():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", select=["b", "d"])

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "b"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["b", "d"]
    assert ds.name_to_index == {"b": 0, "d": 1}

    assert ds.shape == (365 * 4, 2, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "bd")
    slices(ds)


def test_select_2():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", select=["c", "a"])

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

    assert ds.variables == ["c", "a"]
    assert ds.name_to_index == {"c": 0, "a": 1}

    assert ds.shape == (365 * 4, 2, VALUES)

    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "ac")
    slices(ds)


def test_select_3():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", select={"c", "a"})

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "c")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "c"]
    assert ds.name_to_index == {"a": 0, "c": 1}

    assert ds.shape == (365 * 4, 2, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "ac")
    slices(ds)


def test_rename():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", rename={"a": "x", "c": "y"})

    assert isinstance(ds, Rename)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["x", "b", "y", "d"]
    assert ds.name_to_index == {"x": 0, "b": 1, "y": 2, "d": 3}

    assert ds.shape == (365 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "xbyd", "abcd")
    slices(ds)


def test_drop():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", drop="a")

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

    assert ds.variables == ["b", "c", "d"]
    assert ds.name_to_index == {"b": 0, "c": 1, "d": 2}

    assert ds.shape == (365 * 4, 3, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "bcd")
    slices(ds)


def test_reorder_1():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", reorder=["d", "c", "b", "a"])

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "d"), _(date, "c"), _(date, "b"), _(date, "a")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["d", "c", "b", "a"]
    assert ds.name_to_index == {"a": 3, "b": 2, "c": 1, "d": 0}

    assert ds.shape == (365 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_reorder_2():
    ds = open_dataset("test-2021-2021-6h-o96-abcd", reorder=dict(a=3, b=2, c=1, d=0))

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "d"), _(date, "c"), _(date, "b"), _(date, "a")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["d", "c", "b", "a"]
    assert ds.name_to_index == {"a": 3, "b": 2, "c": 1, "d": 0}

    assert ds.shape == (365 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_constructor_1():
    ds1 = open_dataset("test-2021-2021-6h-o96-abcd")

    ds2 = open_dataset("test-2022-2022-6h-o96-abcd")

    ds = open_dataset(ds1, ds2)

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 2 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_constructor_2():
    ds = open_dataset(
        datasets=["test-2021-2021-6h-o96-abcd", "test-2022-2022-6h-o96-abcd"]
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 2 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_constructor_3():
    ds = open_dataset(
        {"datasets": ["test-2021-2021-6h-o96-abcd", "test-2022-2022-6h-o96-abcd"]}
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 2 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_constructor_4():
    ds = open_dataset(
        "test-2021-2021-6h-o96-abcd",
        {"dataset": "test-2022-2022-1h-o96-abcd", "frequency": 6},
    )

    assert isinstance(ds, Concat)
    assert len(ds) == 365 * 2 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [_(date, "a"), _(date, "b"), _(date, "c"), _(date, "d")]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 2 * 4, 4, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_constructor_5():
    ds = open_dataset(
        {"dataset": "test-2021-2021-6h-o96-abcd-1", "rename": {"a": "x", "c": "y"}},
        {"dataset": "test-2021-2021-6h-o96-abcd-2", "rename": {"c": "z", "d": "t"}},
    )

    assert isinstance(ds, Select)
    assert len(ds) == 365 * 4
    assert len([row for row in ds]) == len(ds)

    print(ds.variables)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            _(date, "a", 1),
            _(date, "b", 2),
            _(date, "c", 1),
            _(date, "d", 1),
            _(date, "a", 2),
            _(date, "c", 2),
            _(date, "d", 2),
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["x", "b", "y", "d", "a", "z", "t"]
    assert ds.name_to_index == {"x": 0, "b": 1, "y": 2, "d": 3, "a": 4, "z": 5, "t": 6}

    assert ds.shape == (365 * 4, 7, VALUES)
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd-1"), "xyd", "acd")
    same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd-2"), "abzt", "abcd")
    slices(ds)


def test_dates():
    assert _as_first_date(2021) == np.datetime64("2021-01-01T00:00:00")
    assert _as_last_date(2021) == np.datetime64("2021-12-31T23:59:59")
    assert _as_first_date("2021") == np.datetime64("2021-01-01T00:00:00")
    assert _as_last_date("2021") == np.datetime64("2021-12-31T23:59:59")

    assert _as_first_date(202106) == np.datetime64("2021-06-01T00:00:00")
    assert _as_last_date(202108) == np.datetime64("2021-08-31T23:59:59")
    assert _as_first_date("202106") == np.datetime64("2021-06-01T00:00:00")
    assert _as_last_date("202108") == np.datetime64("2021-08-31T23:59:59")
    assert _as_first_date("2021-06") == np.datetime64("2021-06-01T00:00:00")
    assert _as_last_date("2021-08") == np.datetime64("2021-08-31T23:59:59")

    assert _as_first_date(20210101) == np.datetime64("2021-01-01T00:00:00")
    assert _as_last_date(20210101) == np.datetime64("2021-01-01T23:59:59")
    assert _as_first_date("20210101") == np.datetime64("2021-01-01T00:00:00")
    assert _as_last_date("20210101") == np.datetime64("2021-01-01T23:59:59")
    assert _as_first_date("2021-01-01") == np.datetime64("2021-01-01T00:00:00")
    assert _as_last_date("2021-01-01") == np.datetime64("2021-01-01T23:59:59")


def test_slice_1():
    ds = open_dataset("test-2021-2021-6h-o96-abcd")

    s = ds[0:10:2]
    assert len(s) == 5
    assert (s[0] == ds[0]).all()
    assert (s[1] == ds[2]).all()
    assert (s[2] == ds[4]).all()
    assert (s[3] == ds[6]).all()
    assert (s[4] == ds[8]).all()


def test_slice_2():
    ds = open_dataset("test-2021-2021-6h-o96-abcd")
    s = ds[2:5]
    assert len(s) == 3
    assert (s[0] == ds[2]).all()
    assert (s[1] == ds[3]).all()
    assert (s[2] == ds[5]).all()


def test_slice_3():
    ds = open_dataset("test-2021-2021-6h-o96-abcd")

    s = ds[:5:2]
    assert len(s) == 3
    assert (s[0] == ds[0]).all()
    assert (s[1] == ds[2]).all()
    assert (s[2] == ds[4]).all()


def test_slice_4():
    ds = open_dataset("test-2021-2021-6h-o96-abcd")

    n = len(ds)
    s = ds[2:]
    assert len(s) == n - 2
    assert (s[0] == ds[2]).all()
    assert (s[n - 3] == ds[n - 1]).all()


def test_slice_5():
    ds = open_dataset("test-2021-2021-6h-o96-abcd")

    n = len(ds)
    s = ds[2 : (n + 10)]  # slice too large
    assert len(s) == n - 2
    assert (s[0] == ds[2]).all()
    assert (s[n - 3] == ds[n - 1]).all()


def test_slice_6():
    ds = open_dataset([f"test-{year}-{year}-1h-o96-abcd" for year in range(1940, 2023)])

    slices(ds)
    slices(ds, 0, len(ds), 1)
    slices(ds, 0, len(ds), 10)
    slices(ds, 7, -123, 13)

    for i in range(0, len(ds), len(ds) // 10):
        for s in range(1, 3):
            slices(ds, i, i + s, 1)
            slices(ds, i, i + s, 10)
            slices(ds, i, i + s, 13)


def test_slice_7():
    ds = open_dataset(
        [
            f"test-2020-2020-1h-o96-{vars}"
            for vars in ("abcd", "efgh", "ijkl", "mnop", "qrst", "uvwx", "yz")
        ]
    )

    slices(ds)
    slices(ds, 0, len(ds), 1)
    slices(ds, 0, len(ds), 10)
    slices(ds, 7, -123, 13)

    for i in range(0, len(ds), len(ds) // 10):
        for s in range(1, 3):
            slices(ds, i, i + s, 1)
            slices(ds, i, i + s, 10)
            slices(ds, i, i + s, 13)


def test_slice_8():
    ds = open_dataset(
        [f"test-2020-2020-1h-o96-{vars}" for vars in ("abcd", "cd", "a", "c")]
    )

    slices(ds)
    slices(ds, 0, len(ds), 1)
    slices(ds, 0, len(ds), 10)
    slices(ds, 7, -123, 13)

    for i in range(0, len(ds), len(ds) // 10):
        for s in range(1, 3):
            slices(ds, i, i + s, 1)
            slices(ds, i, i + s, 10)
            slices(ds, i, i + s, 13)


def test_slice_9():
    ds = open_dataset(
        [f"test-{year}-{year}-1h-o96-abcd" for year in range(1940, 2023)],
        frequency=18,
    )

    slices(ds)
    slices(ds, 0, len(ds), 1)
    slices(ds, 0, len(ds), 10)
    slices(ds, 7, -123, 13)

    for i in range(0, len(ds), len(ds) // 10):
        for s in range(1, 3):
            slices(ds, i, i + s, 1)
            slices(ds, i, i + s, 10)
            slices(ds, i, i + s, 13)


def test_ensemble_1():
    ds = open_dataset(
        ensemble=[
            "test-2021-2021-6h-o96-abcd-1-10",
            "test-2021-2021-6h-o96-abcd-2-1",
        ]
    )

    assert isinstance(ds, Ensemble)
    assert len(ds) == 365 * 1 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            [_(date, "a", 1, i) for i in range(10)] + [_(date, "a", 2, 0)],
            [_(date, "b", 1, i) for i in range(10)] + [_(date, "b", 2, 0)],
            [_(date, "c", 1, i) for i in range(10)] + [_(date, "c", 2, 0)],
            [_(date, "d", 1, i) for i in range(10)] + [_(date, "d", 2, 0)],
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 4, 4, 11, VALUES)
    # same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


def test_ensemble_2():
    ds = open_dataset(
        ensemble=[
            "test-2021-2021-6h-o96-abcd-1-10",
            "test-2021-2021-6h-o96-abcd-2-1",
            "test-2021-2021-6h-o96-abcd-3-5",
        ]
    )

    assert isinstance(ds, Ensemble)
    assert len(ds) == 365 * 1 * 4
    assert len([row for row in ds]) == len(ds)

    dates = []
    date = datetime.datetime(2021, 1, 1)

    for row in ds:
        expect = [
            [_(date, "a", 1, i) for i in range(10)]
            + [_(date, "a", 2, 0)]
            + [_(date, "a", 3, i) for i in range(5)],
            [_(date, "b", 1, i) for i in range(10)]
            + [_(date, "b", 2, 0)]
            + [_(date, "b", 3, i) for i in range(5)],
            [_(date, "c", 1, i) for i in range(10)]
            + [_(date, "c", 2, 0)]
            + [_(date, "c", 3, i) for i in range(5)],
            [_(date, "d", 1, i) for i in range(10)]
            + [_(date, "d", 2, 0)]
            + [_(date, "d", 3, i) for i in range(5)],
        ]
        assert (row == expect).all()
        dates.append(date)
        date += datetime.timedelta(hours=6)

    assert (ds.dates == np.array(dates, dtype="datetime64")).all()

    assert ds.variables == ["a", "b", "c", "d"]
    assert ds.name_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}

    assert ds.shape == (365 * 4, 4, 16, VALUES)
    # same_stats(ds, open_dataset("test-2021-2021-6h-o96-abcd"), "abcd")
    slices(ds)


if __name__ == "__main__":
    test_ensemble_1()