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

        return [i for i, date in enumerate(self.dates) if start <= date.year <= end]


class Dataset(Base):
    def __init__(self, path):
        print(path)
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


class Concat(Base):
    def __init__(self, *datasets):
        self.datasets = list(datasets)

        # TODO: check that all datasets are compatible

    def __len__(self):
        return sum(len(i) for i in self.datasets)

    def __getitem__(self, n):
        # TODO: optimize
        k = 0
        while n >= len(self.datasets[k]):
            n -= len(self.datasets[k])
            k += 1
        return self.datasets[k][n]


class Subset(Base):
    def __init__(self, dataset, indices):
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


def open_dataset(*args, **kwargs):
    paths = [name_to_path(name) for name in args]
    assert len(paths) >= 1
    if len(paths) > 2:
        return Concat([Dataset(path) for path in paths]).subset(**kwargs)
    return Dataset(paths[0]).subset(**kwargs)
