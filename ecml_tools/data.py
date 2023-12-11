# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
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
from functools import cached_property
from pathlib import PurePath

import numpy as np
import yaml
import zarr

LOG = logging.getLogger(__name__)

__all__ = ["open_dataset", "open_zarr"]


class Dataset:
    arguments = {}

    @cached_property
    def _len(self):
        return len(self)

    def _subset(self, **kwargs):
        if not kwargs:
            return self

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            end = kwargs.pop("end", None)

            return Subset(self, self._dates_to_indices(start, end))._subset(**kwargs)

        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency")
            return Subset(self, self._frequency_to_indices(frequency))._subset(**kwargs)

        if "select" in kwargs:
            select = kwargs.pop("select")
            return Select(self, self._select_to_columns(select))._subset(**kwargs)

        if "drop" in kwargs:
            drop = kwargs.pop("drop")
            return Select(self, self._drop_to_columns(drop))._subset(**kwargs)

        if "reorder" in kwargs:
            reorder = kwargs.pop("reorder")
            return Select(self, self._reorder_to_columns(reorder))._subset(**kwargs)

        if "rename" in kwargs:
            rename = kwargs.pop("rename")
            return Rename(self, rename)._subset(**kwargs)

        if "statistics" in kwargs:
            statistics = kwargs.pop("statistics")
            return Statistics(self, statistics)._subset(**kwargs)

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def _frequency_to_indices(self, frequency):
        requested_frequency = _frequency_to_hours(frequency)
        dataset_frequency = _frequency_to_hours(self.frequency)
        assert requested_frequency % dataset_frequency == 0
        # Question: where do we start? first date, or first date that is a multiple of the frequency?
        step = requested_frequency // dataset_frequency

        return range(0, len(self), step)

    def _dates_to_indices(self, start, end):
        # TODO: optimize

        start = self.dates[0] if start is None else _as_first_date(start, self.dates)
        end = self.dates[-1] if end is None else _as_last_date(end, self.dates)

        return [i for i, date in enumerate(self.dates) if start <= date <= end]

    def _select_to_columns(self, vars):
        if isinstance(vars, set):
            # We keep the order of the variables as they are in the zarr file
            nvars = [v for v in self.name_to_index if v in vars]
            assert len(nvars) == len(vars)
            return self._select_to_columns(nvars)

        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        return [self.name_to_index[v] for v in vars]

    def _drop_to_columns(self, vars):
        if not isinstance(vars, (list, tuple, set)):
            vars = [vars]

        assert set(vars) <= set(self.name_to_index)

        return sorted([v for k, v in self.name_to_index.items() if k not in vars])

    def _reorder_to_columns(self, vars):
        if isinstance(vars, (list, tuple)):
            vars = {k: i for i, k in enumerate(vars)}

        indices = []
        for k, v in sorted(self.name_to_index.items(), key=lambda x: x[1]):
            indices.append(vars[k])

        assert set(indices) == set(range(len(self.name_to_index)))

        return indices

    def dates_interval_to_indices(self, start, end):
        return self._dates_to_indices(start, end)

    def metadata(self):
        raise NotImplementedError()

    def provenance(self):
        return {}

    def sub_shape(self, drop_axis):
        shape = self.shape
        shape = list(shape)
        shape.pop(drop_axis)
        return tuple(shape)


class ReadOnlyStore(zarr.storage.BaseStore):
    def __delitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class HTTPStore(ReadOnlyStore):
    """
    We write our own HTTPStore because the one used by zarr (fsspec)
    does not play well with fork() and multiprocessing.
    """

    def __init__(self, url):
        self.url = url

    def __getitem__(self, key):
        import requests

        r = requests.get(self.url + "/" + key)

        if r.status_code == 404:
            raise KeyError(key)

        r.raise_for_status()
        return r.content


class S3Store(ReadOnlyStore):
    """
    We write our own S3Store because the one used by zarr (fsspec)
    does not play well with fork() and multiprocessing.
    """

    def __init__(self, url):
        import boto3

        self.bucket, self.key = url[5:].split("/", 1)

        # TODO: get the profile name from the url
        self.s3 = boto3.Session(profile_name=None).client("s3")

    def __getitem__(self, key):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key + "/" + key)
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(key)

        return response["Body"].read()


def open_zarr(path):
    store = path
    if store.startswith("http://") or store.startswith("https://"):
        store = HTTPStore(store)

    elif store.startswith("s3://"):
        store = S3Store(store)

    return zarr.convenience.open(store, "r")


class Zarr(Dataset):
    def __init__(self, path):
        if isinstance(path, zarr.hierarchy.Group):
            self.path = str(id(path))
            self.z = path
        else:
            self.path = str(path)
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z.data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, n):
        return self.data[n]

    @cached_property
    def shape(self):
        return self.data.shape

    @cached_property
    def dtype(self):
        return self.data.dtype

    @cached_property
    def dates(self):
        return self.z.dates[:]  # Convert to numpy

    @property
    def latitudes(self):
        try:
            return self.z.latitudes[:]
        except AttributeError:
            LOG.warning("No 'latitudes' in %r, trying 'latitude'", self)
            return self.z.latitude[:]

    @property
    def longitudes(self):
        try:
            return self.z.longitudes[:]
        except AttributeError:
            LOG.warning("No 'longitudes' in %r, trying 'longitude'", self)
            return self.z.longitude[:]

    @property
    def statistics(self):
        return dict(
            mean=self.z.mean[:],
            stdev=self.z.stdev[:],
            maximum=self.z.maximum[:],
            minimum=self.z.minimum[:],
        )

    @property
    def resolution(self):
        return self.z.attrs["resolution"]

    @property
    def frequency(self):
        try:
            return self.z.attrs["frequency"]
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'dates'", self)
        dates = self.dates
        delta = dates[1].astype(object) - dates[0].astype(object)
        return delta.total_seconds() // 3600

    @property
    def name_to_index(self):
        if "variables" in self.z.attrs:
            return {n: i for i, n in enumerate(self.z.attrs["variables"])}
        return self.z.attrs["name_to_index"]

    @property
    def variables(self):
        return [
            k
            for k, v in sorted(
                self.name_to_index.items(),
                key=lambda x: x[1],
            )
        ]

    def metadata(self):
        result = {}
        for k, v in self.z.attrs.items():
            if not k.startswith("_"):
                result[k] = v
        return result

    def __repr__(self):
        return self.path

    def end_of_statistics_date(self):
        return self.dates[-1]


class Forwards(Dataset):
    def __init__(self, forward):
        self.forward = forward

    def __len__(self):
        return len(self.forward)

    def __getitem__(self, n):
        return self.forward[n]

    @property
    def dates(self):
        return self.forward.dates

    @property
    def resolution(self):
        return self.forward.resolution

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def latitudes(self):
        return self.forward.latitudes

    @property
    def longitudes(self):
        return self.forward.longitudes

    @property
    def name_to_index(self):
        return self.forward.name_to_index

    @property
    def variables(self):
        return self.forward.variables

    @property
    def statistics(self):
        return self.forward.statistics

    @property
    def shape(self):
        return self.forward.shape

    @property
    def dtype(self):
        return self.forward.dtype

    def metadata(self):
        return self.forward.metadata()


class Combined(Forwards):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

        # Forward most properties to the first dataset
        super().__init__(datasets[0])

    def check_same_resolution(self, d1, d2):
        if d1.resolution != d2.resolution:
            raise ValueError(
                f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})"
            )

    def check_same_frequency(self, d1, d2):
        if d1.frequency != d2.frequency:
            raise ValueError(
                f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})"
            )

    def check_same_grid(self, d1, d2):
        if (d1.latitudes != d2.latitudes).any() or (
            d1.longitudes != d2.longitudes
        ).any():
            raise ValueError(f"Incompatible grid ({d1} {d2})")

    def check_same_shape(self, d1, d2):
        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

        if d1.variables != d2.variables:
            raise ValueError(
                f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})"
            )

    def check_same_sub_shapes(self, d1, d2, drop_axis):
        shape1 = d1.sub_shape(drop_axis)
        shape2 = d2.sub_shape(drop_axis)

        if shape1 != shape2:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

    def check_same_variables(self, d1, d2):
        if d1.variables != d2.variables:
            raise ValueError(
                f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})"
            )

    def check_same_lengths(self, d1, d2):
        if d1._len != d2._len:
            raise ValueError(f"Incompatible lengths: {d1._len} and {d2._len}")

    def check_same_dates(self, d1, d2):
        self.check_same_frequency(d1, d2)

        if d1.dates[0] != d2.dates[0]:
            raise ValueError(
                f"Incompatible start dates: {d1.dates[0]} and {d2.dates[0]} ({d1} {d2})"
            )

        if d1.dates[-1] != d2.dates[-1]:
            raise ValueError(
                f"Incompatible end dates: {d1.dates[-1]} and {d2.dates[-1]} ({d1} {d2})"
            )

    def check_compatibility(self, d1, d2):
        # These are the default checks
        # Derived classes should turn individual checks off if they are not needed
        self.check_same_resolution(d1, d2)
        self.check_same_frequency(d1, d2)
        self.check_same_grid(d1, d2)
        self.check_same_lengths(d1, d2)
        self.check_same_variables(d1, d2)
        self.check_same_dates(d1, d2)

    def provenance(self):
        result = {}
        for d in self.datasets:
            result.update(d.provenance())
        return result

    def __repr__(self):
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"{self.__class__.__name__}({lst})"


class Concat(Combined):
    def __len__(self):
        return sum(len(i) for i in self.datasets)

    def __getitem__(self, n):
        if isinstance(n, slice):
            return self._get_slice(n)

        # TODO: optimize
        k = 0
        while n >= self.datasets[k]._len:
            n -= self.datasets[k]._len
            k += 1
        return self.datasets[k][n]

    def _get_slice(self, s):
        result = []

        start, stop, step = s.indices(self._len)

        for d in self.datasets:
            length = d._len

            result.append(d[start:stop:step])

            start -= length
            while start < 0:
                start += step

            stop -= length

            if start > stop:
                break

        return np.concatenate(result)

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def check_same_lengths(self, d1, d2):
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1, d2):
        # Turned off because we are concatenating along the dates axis
        pass

    @property
    def dates(self):
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self):
        return (len(self),) + self.datasets[0].shape[1:]


class GivenAxis(Combined):
    """Given a given axis, combine the datasets along that axis"""

    def __init__(self, datasets, axis):
        self.axis = axis
        super().__init__(datasets)

        assert axis > 0 and axis < len(self.datasets[0].shape), (
            axis,
            self.datasets[0].shape,
        )

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=self.axis)

    def metadata(self):
        raise NotImplementedError()

    @cached_property
    def shape(self):
        shapes = [d.shape for d in self.datasets]
        before = shapes[0][: self.axis]
        after = shapes[0][self.axis + 1 :]
        result = before + (sum(s[self.axis] for s in shapes),) + after
        assert False not in result, result
        return result

    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    def __getitem__(self, n):
        if isinstance(n, slice):
            return self._get_slice(n)
        return np.concatenate([d[n] for d in self.datasets], axis=self.axis - 1)


class Ensemble(GivenAxis):
    pass


class Grids(GivenAxis):
    def __init__(self, datasets, axis):
        super().__init__(datasets, axis)
        # Shape: (dates, variables, ensemble, 1d-values)
        assert len(datasets[0].shape) == 4, "Grids must be 1D for now"

    # TODO: select the statistics of the most global grid?
    @property
    def latitudes(self):
        return np.concatenate([d.latitudes for d in self.datasets])

    @property
    def longitudes(self):
        return np.concatenate([d.longitudes for d in self.datasets])

    def check_same_grid(self, d1, d2):
        # We don't check the grid, because we want to be able to combine
        pass


class Join(Combined):
    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=1)

    def check_same_variables(self, d1, d2):
        # Turned off because we are joining along the variables axis
        pass

    def __len__(self):
        return len(self.datasets[0])

    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    def __getitem__(self, n):
        if isinstance(n, slice):
            return self._get_slice(n)
        return np.concatenate([d[n] for d in self.datasets])

    @cached_property
    def shape(self):
        cols = sum(d.shape[1] for d in self.datasets)
        return (len(self), cols) + self.datasets[0].shape[2:]

    def _overlay(self):
        indices = {}
        i = 0
        for d in self.datasets:
            for v in d.variables:
                indices[v] = i
                i += 1

        if len(indices) == i:
            # No overlay
            return self

        indices = list(indices.values())

        i = 0
        for d in self.datasets:
            ok = False
            for v in d.variables:
                if i in indices:
                    ok = True
                i += 1
            if not ok:
                LOG.warning("Dataset %r completly occulted.", d)

        return Select(self, indices)

    @cached_property
    def variables(self):
        seen = set()
        result = []
        for d in reversed(self.datasets):
            for v in reversed(d.variables):
                while v in seen:
                    v = f"({v})"
                seen.add(v)
                result.insert(0, v)

        return result

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    @property
    def statistics(self):
        return {
            k: np.concatenate([d.statistics[k] for d in self.datasets], axis=0)
            for k in self.datasets[0].statistics
        }


class Subset(Forwards):
    def __init__(self, dataset, indices):
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)

        # Forward other properties to the super dataset
        super().__init__(dataset)

    def __getitem__(self, n):
        if isinstance(n, slice):
            return self._get_slice(n)
        n = self.indices[n]
        return self.dataset[n]

    def _get_slice(self, s):
        # TODO: check if the indices can be simplified to a slice
        # the time checking maybe be longer than the time saved
        # using a slice
        indices = [self.indices[i] for i in range(*s.indices(self._len))]
        return np.stack([self.dataset[i] for i in indices])

    def __len__(self):
        return len(self.indices)

    @cached_property
    def shape(self):
        return (len(self),) + self.dataset.shape[1:]

    @cached_property
    def dates(self):
        return self.dataset.dates[self.indices]

    @cached_property
    def frequency(self):
        dates = self.dates
        delta = dates[1].astype(object) - dates[0].astype(object)
        return delta.total_seconds() // 3600


class Select(Forwards):
    def __init__(self, dataset, indices):
        while isinstance(dataset, Select):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)
        assert len(self.indices) > 0

        # Forward other properties to the main dataset
        super().__init__(dataset)

    def __getitem__(self, n):
        row = self.dataset[n]
        if isinstance(n, slice):
            return row[:, self.indices]
        return row[self.indices]

    @cached_property
    def shape(self):
        return (len(self), len(self.indices)) + self.dataset.shape[2:]

    @cached_property
    def variables(self):
        return [self.dataset.variables[i] for i in self.indices]

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    @cached_property
    def statistics(self):
        return {k: v[self.indices] for k, v in self.dataset.statistics.items()}


class Rename(Forwards):
    def __init__(self, dataset, rename):
        super().__init__(dataset)
        for n in rename:
            assert n in dataset.variables
        self._variables = [rename.get(v, v) for v in dataset.variables]

    @property
    def variables(self):
        return self._variables

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}


class Statistics(Forwards):
    def __init__(self, dataset, statistic):
        super().__init__(dataset)
        self._statistic = statistic

    @cached_property
    def statistics(self):
        return open_dataset(self._statistic).statistics


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
        assert d.minutes == 0 and d.hours == 0 and d.seconds == 0, d
        d = datetime.date(d.year, d.month, d.day)

    if isinstance(d, datetime.date):
        d = d.year * 10_000 + d.month * 100 + d.day

    try:
        d = int(d)
    except ValueError:
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


def _concat_or_join(datasets):
    # Study the dates
    ranges = [(d.dates[0].astype(object), d.dates[-1].astype(object)) for d in datasets]

    if len(set(ranges)) == 1:
        return Join(datasets)._overlay()

    # Make sure the dates are disjoint
    for i in range(len(ranges)):
        r = ranges[i]
        for j in range(i + 1, len(ranges)):
            s = ranges[j]
            if r[0] <= s[0] <= r[1] or r[0] <= s[1] <= r[1]:
                raise ValueError(
                    f"Overlapping dates: {r} and {s} ({datasets[i]} {datasets[j]})"
                )

    # For now we should have the datasets in order with no gaps

    frequency = _frequency_to_hours(datasets[0].frequency)

    for i in range(len(ranges) - 1):
        r = ranges[i]
        s = ranges[i + 1]
        if r[1] + datetime.timedelta(hours=frequency) != s[0]:
            raise ValueError(
                "Datasets must be sorted by dates, with no gaps: "
                f"{r} and {s} ({datasets[i]} {datasets[i+1]})"
            )

    return Concat(datasets)


def _open(a, zarr_root):
    if isinstance(a, Dataset):
        return a

    if isinstance(a, zarr.hierarchy.Group):
        return Zarr(a)

    if isinstance(a, str):
        return Zarr(_name_to_path(a, zarr_root))

    if isinstance(a, PurePath):
        return Zarr(str(a))

    if isinstance(a, dict):
        return _open_dataset(zarr_root=zarr_root, **a)

    if isinstance(a, (list, tuple)):
        return _open_dataset(*a, zarr_root=zarr_root)

    raise NotImplementedError()


def _open_dataset(*args, zarr_root, **kwargs):
    sets = []
    for a in args:
        sets.append(_open(a, zarr_root))

    if "ensemble" in kwargs:
        if "grids" in kwargs:
            raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")
        ensemble = kwargs.pop("ensemble")
        axis = kwargs.pop("axis", 2)
        assert len(args) == 0
        assert isinstance(ensemble, (list, tuple))
        return Ensemble([_open(e, zarr_root) for e in ensemble], axis=axis)._subset(
            **kwargs
        )

    if "grids" in kwargs:
        if "ensemble" in kwargs:
            raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")
        grids = kwargs.pop("grids")
        axis = kwargs.pop("axis", 3)
        assert len(args) == 0
        assert isinstance(grids, (list, tuple))
        return Grids([_open(e, zarr_root) for e in grids], axis=axis)._subset(**kwargs)

    for name in ("datasets", "dataset"):
        if name in kwargs:
            datasets = kwargs.pop(name)
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            for a in datasets:
                sets.append(_open(a, zarr_root))

    assert len(sets) > 0, (args, kwargs)

    if len(sets) > 1:
        return _concat_or_join(sets)._subset(**kwargs)

    return sets[0]._subset(**kwargs)


def open_dataset(*args, zarr_root=None, **kwargs):
    ds = _open_dataset(*args, zarr_root=zarr_root, **kwargs)
    ds.arguments = {"args": args, "kwargs": kwargs}
    return ds
