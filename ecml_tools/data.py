# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import re
from functools import cached_property

import yaml
import zarr

# Question: should properties be cached?


class Base:
    def subset(self, **kwargs):
        if not kwargs:
            return self

        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency")

            return Subset(self, self.frequency_to_indices(frequency)).subset(**kwargs)

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start")
            end = kwargs.pop("end")

            def is_year(x):
                return isinstance(x, int) and 1000 <= x <= 9999

            if start is None or is_year(start):
                if end is None or is_year(end):
                    return Subset(self, self.years_to_indices(start, end)).subset(
                        **kwargs
                    )

            raise NotImplementedError(f"Unsupported start/end: {start} {end}")

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def frequency_to_indices(self, frequency):
        requested_frequency = _frequency_to_hours(frequency)
        delta = self.dates[1] - self.dates[0]
        dataset_frequency = delta.item().total_seconds() / 3600
        assert dataset_frequency.is_integer()
        dataset_frequency = int(dataset_frequency)
        assert requested_frequency % dataset_frequency == 0
        # Question: where do we start? first date, or first date that is a multiple of the frequency?
        step = requested_frequency // dataset_frequency

        return range(0, len(self), step)

    def years_to_indices(self, start, end):
        # TODO: optimize
        start = self.dates[0].year if start is None else start
        end = self.dates[-1].year if end is None else end

        return [
            i
            for i, date in enumerate(self.dates)
            if start <= date.astype(object).year <= end
        ]


class Dataset(Base):
    def __init__(self, path):
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

    @cached_property
    def resolution(self):
        return self.z.attrs["resolution"]

    @cached_property
    def frequency(self):
        return self.z.attrs["frequency"]


class Concat(Base):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

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
        # TODO:
        # - check dates
        # - check parameters
        if d1.resolution != d2.resolution:
            raise ValueError(
                f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})"
            )

        if d1.frequency != d2.frequency:
            raise ValueError(
                f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})"
            )

        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )


class Join(Base):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

    def check_compatibility(self, d1, d2):
        pass

    @property
    def dates(self):
        return self.datasets[0].dates

    @property
    def resolution(self):
        return self.datasets[0].resolution

    @property
    def frequency(self):
        return self.datasets[0].frequency


class Subset(Base):
    def __init__(self, dataset, indices):
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)

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


def name_to_path(name):
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


def concat_or_join(datasets):
    # Study the dates
    ranges = [(d.dates[0], d.dates[-1]) for d in datasets]

    if len(set(ranges)) == 1:
        return Join(datasets)

    # Make sure the dates are disjoint
    for i in range(len(ranges)):
        r = ranges[i]
        for j in range(i + 1, len(ranges)):
            s = ranges[j]
            if r[0] <= s[0] <= r[1] or r[0] <= s[1] <= r[1]:
                raise ValueError(
                    f"Overlapping dates: {r} and {s} ({datasets[i]} {datasets[j]})"
                )

    # TODO:
    # - should we sort the datasets by date?
    # - should we check for gaps?
    return Concat(datasets)


def open_dataset(*args, **kwargs):
    paths = [name_to_path(name) for name in args]
    assert len(paths) >= 1
    if len(paths) > 1:
        return concat_or_join([Dataset(path) for path in paths]).subset(**kwargs)
    return Dataset(paths[0]).subset(**kwargs)
