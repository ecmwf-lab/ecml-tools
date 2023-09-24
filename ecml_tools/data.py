# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import calendar
import datetime
import json
import logging
import os
import re
from functools import cached_property

import numpy as np
import tqdm
import yaml
import zarr

LOG = logging.getLogger(__name__)

__all__ = ["open_dataset"]


class Dataset:
    metadata = {}

    @cached_property
    def _len(self):
        return len(self)

    def _subset(self, **kwargs):
        if not kwargs:
            return self

        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency")

            return Subset(self, self._frequency_to_indices(frequency))._subset(**kwargs)

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            end = kwargs.pop("end", None)

            return Subset(self, self._dates_to_indices(start, end))._subset(**kwargs)

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

        start = self.dates[0] if start is None else _as_first_date(start)
        end = self.dates[-1] if end is None else _as_last_date(end)

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

    def save(self, path, chunks=None, buffer_size=10):
        z = zarr.convenience.open(path, "w")

        print(json.dumps(self.metadata, indent=4))
        print(json.dumps(self.zarrs(), indent=4))

        if chunks is None:
            chunks = (1,) + self.shape[1:]

        z.create_dataset(
            "data",
            shape=self.shape,
            chunks=chunks,
            dtype=self.dtype,
        )

        z.create_dataset("dates", data=self.dates)
        z.create_dataset("latitudes", data=self.latitudes)
        z.create_dataset("longitudes", data=self.longitudes)

        for k, v in self.statistics.items():
            z.create_dataset(k, data=v)

        z.attrs["resolution"] = self.resolution
        z.attrs["frequency"] = self.frequency
        z.attrs["name_to_index"] = self.name_to_index
        z.attrs["variables"] = self.variables

        for i in tqdm.tqdm(range(0, self._len, buffer_size)):
            j = min(i + buffer_size, self._len)
            z.data[i:j] = self[i:j]

    def zarrs(self):
        return {}


class Zarr(Dataset):
    def __init__(self, path):
        if isinstance(path, zarr.hierarchy.Group):
            self.path = str(id(path))
            self.z = path
        else:
            self.path = path
            self.z = zarr.convenience.open(path, "r")

    def __len__(self):
        return self.z.data.shape[0]

    def __getitem__(self, n):
        return self.z.data[n]

    @property
    def shape(self):
        return self.z.data.shape

    @property
    def dtype(self):
        return self.z.data.dtype

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

    def zarrs(self):
        return {self.path: dict(**self.z.attrs)}

    def __repr__(self):
        return self.path


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

    def zarrs(self):
        return self.forward.zarrs()


class Combined(Forwards):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

        # Forward most properties to the first dataset
        super().__init__(datasets[0])

    def check_compatibility(self, d1, d2):
        if d1.resolution != d2.resolution:
            raise ValueError(
                f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})"
            )

        if d1.frequency != d2.frequency:
            raise ValueError(
                f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})"
            )

        if (d1.latitudes != d2.latitudes).any() or (
            d1.longitudes != d2.longitudes
        ).any():
            raise ValueError(f"Incompatible grid ({d1} {d2})")

    def zarrs(self):
        result = {}
        for d in self.datasets:
            result.update(d.zarrs())
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

        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

        if d1.variables != d2.variables:
            raise ValueError(
                f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})"
            )

    @property
    def dates(self):
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self):
        return (len(self),) + self.datasets[0].shape[1:]


class Join(Combined):
    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        if (d1.shape[0],) + d1.shape[2:] != (d2.shape[0],) + d2.shape[2:]:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

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


def _name_to_path(name):
    if name.endswith(".zarr"):
        return name

    with open(os.path.expanduser("~/.ecml-tools")) as f:
        config = yaml.safe_load(f)

    return os.path.join(config["zarr_root"], name + ".zarr")


def _frequency_to_hours(frequency):
    if isinstance(frequency, int):
        return frequency

    m = re.match(r"(\d+)([dh])?", frequency)
    if m is None:
        raise ValueError("Invalid frequency: " + frequency)

    frequency = int(m.group(1))
    if m.group(2) == "h":
        return frequency

    if m.group(2) == "d":
        return frequency * 24

    raise NotImplementedError()


def _as_date(d, last):
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
        bits = d.split("-")
        if len(bits) == 1:
            return _as_date(int(bits[0]), last)

        if len(bits) == 2:
            return _as_date(int(bits[0]) * 100 + int(bits[1]), last)

        if len(bits) == 3:
            return _as_date(
                int(bits[0]) * 10000 + int(bits[1]) * 100 + int(bits[2]), last
            )

    raise NotImplementedError(f"Unsupported date: {d} ({type(d)})")


def _as_first_date(d):
    return _as_date(d, last=False)


def _as_last_date(d):
    return _as_date(d, last=True)


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


def _open(a):
    if isinstance(a, Dataset):
        return a

    if isinstance(a, zarr.hierarchy.Group):
        return Zarr(a)

    if isinstance(a, str):
        return Zarr(_name_to_path(a))

    if isinstance(a, dict):
        return _open_dataset(**a)

    if isinstance(a, (list, tuple)):
        return _open_dataset(*a)

    raise NotImplementedError()


def _open_dataset(*args, **kwargs):
    sets = []
    for a in args:
        sets.append(_open(a))

    for name in ("datasets", "dataset"):
        if name in kwargs:
            datasets = kwargs.pop(name)
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            for a in datasets:
                sets.append(_open(a))

    assert len(sets) > 0, (args, kwargs)

    if len(sets) > 1:
        return _concat_or_join(sets)._subset(**kwargs)

    return sets[0]._subset(**kwargs)


def open_dataset(*args, **kwargs):
    ds = _open_dataset(*args, **kwargs)
    ds.metadata = {"args": args, "kwargs": kwargs}
    return ds
