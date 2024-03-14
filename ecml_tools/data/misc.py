# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import calendar
import datetime
import logging
import os
import re
from pathlib import PurePath

import numpy as np
import yaml
import zarr

from .dataset import Dataset

LOG = logging.getLogger(__name__)


def _name_to_path(name, zarr_root):
    _, ext = os.path.splitext(name)

    if ext in (".zarr", ".zip"):
        return name

    if zarr_root is None:
        with open(os.path.expanduser("~/.ecml-tools")) as f:
            zarr_root = yaml.safe_load(f)["zarr_root"]

    return os.path.join(zarr_root, name + ".zarr")


def _frequency_to_hours(frequency):
    if isinstance(frequency, int):
        return frequency

    if isinstance(frequency, float):
        assert int(frequency) == frequency
        return int(frequency)

    m = re.match(r"(\d+)([dh])?", frequency)
    if m is None:
        raise ValueError("Invalid frequency: " + frequency)

    frequency = int(m.group(1))
    if m.group(2) == "h":
        return frequency

    if m.group(2) == "d":
        return frequency * 24

    raise NotImplementedError()


def _as_date(d, dates, last):
    if isinstance(d, datetime.datetime):
        if not d.minute == 0 and d.hour == 0 and d.second == 0:
            return np.datetime64(d)
        d = datetime.date(d.year, d.month, d.day)

    if isinstance(d, datetime.date):
        d = d.year * 10_000 + d.month * 100 + d.day

    try:
        d = int(d)
    except (ValueError, TypeError):
        pass

    if isinstance(d, int):
        if len(str(d)) == 4:
            year = d
            if last:
                return np.datetime64(f"{year:04}-12-31T23:59:59")
            else:
                return np.datetime64(f"{year:04}-01-01T00:00:00")

        if len(str(d)) == 6:
            year = d // 100
            month = d % 100
            if last:
                _, last_day = calendar.monthrange(year, month)
                return np.datetime64(f"{year:04}-{month:02}-{last_day:02}T23:59:59")
            else:
                return np.datetime64(f"{year:04}-{month:02}-01T00:00:00")

        if len(str(d)) == 8:
            year = d // 10000
            month = (d % 10000) // 100
            day = d % 100
            if last:
                return np.datetime64(f"{year:04}-{month:02}-{day:02}T23:59:59")
            else:
                return np.datetime64(f"{year:04}-{month:02}-{day:02}T00:00:00")

    if isinstance(d, str):
        if "-" in d:
            assert ":" not in d
            bits = d.split("-")
            if len(bits) == 1:
                return _as_date(int(bits[0]), dates, last)

            if len(bits) == 2:
                return _as_date(int(bits[0]) * 100 + int(bits[1]), dates, last)

            if len(bits) == 3:
                return _as_date(
                    int(bits[0]) * 10000 + int(bits[1]) * 100 + int(bits[2]),
                    dates,
                    last,
                )
        if ":" in d:
            assert len(d) == 5
            hour, minute = d.split(":")
            assert minute == "00"
            assert not last
            first = dates[0].astype(object)
            year = first.year
            month = first.month
            day = first.day

            return np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour}:00:00")

    raise NotImplementedError(f"Unsupported date: {d} ({type(d)})")


def _as_first_date(d, dates):
    return _as_date(d, dates, last=False)


def _as_last_date(d, dates):
    return _as_date(d, dates, last=True)


def _concat_or_join(datasets, kwargs):
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    # Study the dates
    ranges = [(d.dates[0].astype(object), d.dates[-1].astype(object)) for d in datasets]

    if len(set(ranges)) == 1:
        from .join import Join

        return Join(datasets)._overlay(), kwargs

    # Make sure the dates are disjoint
    for i in range(len(ranges)):
        r = ranges[i]
        for j in range(i + 1, len(ranges)):
            s = ranges[j]
            if r[0] <= s[0] <= r[1] or r[0] <= s[1] <= r[1]:
                raise ValueError(f"Overlapping dates: {r} and {s} ({datasets[i]} {datasets[j]})")

    # For now we should have the datasets in order with no gaps

    frequency = _frequency_to_hours(datasets[0].frequency)

    for i in range(len(ranges) - 1):
        r = ranges[i]
        s = ranges[i + 1]
        if r[1] + datetime.timedelta(hours=frequency) != s[0]:
            raise ValueError(
                "Datasets must be sorted by dates, with no gaps: " f"{r} and {s} ({datasets[i]} {datasets[i+1]})"
            )

    from .concat import Concat

    return Concat(datasets), kwargs


def _open(a, zarr_root):
    from .stores import Zarr

    if isinstance(a, Dataset):
        return a

    if isinstance(a, zarr.hierarchy.Group):
        return Zarr(a).mutate()

    if isinstance(a, str):
        return Zarr(_name_to_path(a, zarr_root)).mutate()

    if isinstance(a, PurePath):
        return Zarr(str(a)).mutate()

    if isinstance(a, dict):
        return _open_dataset(zarr_root=zarr_root, **a)

    if isinstance(a, (list, tuple)):
        return _open_dataset(*a, zarr_root=zarr_root)

    raise NotImplementedError()


def _auto_adjust(datasets, kwargs):
    """Adjust the datasets for concatenation or joining based on parameters set to
    'matching'."""

    if kwargs.get("adjust") == "matching":
        kwargs.pop("adjust")
        for p in ("select", "frequency", "start", "end"):
            kwargs[p] = "matching"

    adjust = {}

    if kwargs.get("select") == "matching":
        kwargs.pop("select")
        variables = None

        for d in datasets:
            if variables is None:
                variables = set(d.variables)
            else:
                variables &= set(d.variables)

        if len(variables) == 0:
            raise ValueError("No common variables")

        adjust["select"] = sorted(variables)

    if kwargs.get("start") == "matching":
        kwargs.pop("start")
        adjust["start"] = max(d.dates[0] for d in datasets).astype(object)

    if kwargs.get("end") == "matching":
        kwargs.pop("end")
        adjust["end"] = min(d.dates[-1] for d in datasets).astype(object)

    if kwargs.get("frequency") == "matching":
        kwargs.pop("frequency")
        adjust["frequency"] = max(d.frequency for d in datasets)

    if adjust:
        datasets = [d._subset(**adjust) for d in datasets]

    return datasets, kwargs


def _open_dataset(*args, zarr_root, **kwargs):
    sets = []
    for a in args:
        sets.append(_open(a, zarr_root))

    if "ensemble" in kwargs:
        from .ensemble import ensemble_factory

        return ensemble_factory(args, kwargs, zarr_root)

    if "grids" in kwargs:
        from .grids import grids_factory

        return grids_factory(args, kwargs, zarr_root)

    for name in ("datasets", "dataset"):
        if name in kwargs:
            datasets = kwargs.pop(name)
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            for a in datasets:
                sets.append(_open(a, zarr_root))

    assert len(sets) > 0, (args, kwargs)

    if len(sets) > 1:
        dataset, kwargs = _concat_or_join(sets, kwargs)
        return dataset._subset(**kwargs)

    return sets[0]._subset(**kwargs)
