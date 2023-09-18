# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import zarr


class Dataset:
    def __init__(self, path):
        self.z = zarr.open(path)

    def __len__(self):
        return self.z.data.shape[0]

    def __getitem__(self, n):
        return self.z.data[n]


class Concat:
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


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, n):
        n = self.indices[n]
        return self.dataset[n]

    def __len__(self):
        return len(self.indices)


def name_to_path(name):
    if name.endswith(".zarr"):
        return name
    # open("~/.ecml-tools")

    return name


def open_dataset(*args, **kwargs):
    paths = [name_to_path(name) for name in args]
    assert len(paths) == 1
    return Dataset(paths[0])
