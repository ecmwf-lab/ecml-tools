# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import re
from functools import cached_property

import numpy as np
import yaml
import zarr

LOG = logging.getLogger(__name__)

__all__ = ["open_dataset"]


class Dataset:
    def _subset(self, **kwargs):
        if not kwargs:
            return self

        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency")

            return Subset(self, self._frequency_to_indices(frequency))._subset(**kwargs)

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start")
            end = kwargs.pop("end")

            def is_year(x):
                return isinstance(x, int) and 1000 <= x <= 9999

            if start is None or is_year(start):
                if end is None or is_year(end):
                    return Subset(self, self._years_to_indices(start, end))._subset(
                        **kwargs
                    )

            raise NotImplementedError(f"Unsupported start/end: {start} {end}")

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

    def _years_to_indices(self, start, end):
        # TODO: optimize
        start = self.dates[0].astype(object).year if start is None else start
        end = self.dates[-1].astype(object).year if end is None else end

        return [
            i
            for i, date in enumerate(self.dates)
            if start <= date.astype(object).year <= end
        ]

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


class Zarr(Dataset):
    def __init__(self, path):
        if isinstance(path, zarr.hierarchy.Group):
            self.path = "-"
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


class Combined(Forwards):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

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


class Concat(Combined):
    def __len__(self):
        return sum(len(i) for i in self.datasets)

    def __getitem__(self, n):
        # TODO: optimize
        k = 0
        while n >= len(self.datasets[k]):
            n -= len(self.datasets[k])
            k += 1
        return self.datasets[k][n]

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

    def __repr__(self):
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"Concat({lst})"

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

    def __getitem__(self, n):
        return np.concatenate([d[n] for d in self.datasets], axis=0)

    @cached_property
    def shape(self):
        cols = sum(d.shape[1] for d in self.datasets)
        return (len(self), cols) + self.datasets[0].shape[2:]

    def __repr__(self):
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"Join({lst})"

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

        super().__init__(dataset)

    def __getitem__(self, n):
        n = self.indices[n]
        return self.dataset[n]

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

        super().__init__(dataset)

    def __getitem__(self, n):
        row = self.dataset[n]
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
        return open_dataset(**a)

    if isinstance(a, (list, tuple)):
        return open_dataset(*a)

    raise NotImplementedError()


def open_dataset(*args, **kwargs):
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
