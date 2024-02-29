# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging
import time
import warnings

import numpy as np

from .check import check_data_values
from .statistics import compute_statistics
from .utils import progress_bar, seconds

LOG = logging.getLogger(__name__)


class CubesFilter:
    def __init__(self, *, parts, total):
        if parts is None:
            self.parts = None
            return

        if len(parts) == 1:
            part = parts[0]
            if part.lower() in ["all", "*"]:
                self.parts = None
                return

            if "/" in part:
                i_chunk, n_chunks = part.split("/")
                i_chunk, n_chunks = int(i_chunk), int(n_chunks)

                assert i_chunk > 0, f"Chunk number {i_chunk} must be positive."
                if n_chunks > total:
                    warnings.warn(
                        f"Number of chunks {n_chunks} is larger than the total number of chunks: {total}+1. "
                        "Some chunks will be empty."
                    )

                chunk_size = total / n_chunks
                parts = [x for x in range(total) if x >= (i_chunk - 1) * chunk_size and x < i_chunk * chunk_size]

        parts = [int(_) for _ in parts]
        LOG.info(f"Running parts: {parts}")
        if not parts:
            warnings.warn(f"Nothing to do for chunk {i_chunk}/{n_chunks}.")

        self.parts = parts

    def __call__(self, i):
        if self.parts is None:
            return True
        return i in self.parts


class ViewCacheArray:
    """A class that provides a caching mechanism for writing to a NumPy-like array.

    The is initialized with a NumPy-like array, a shape and a list to reindex the first dimension.
    The array is used to store the final data, while the cache is used to temporarily
    store the data before flushing it to the array.

    The `flush` method copies the contents of the cache to the final array.
    """

    def __init__(self, array, *, shape, indexes):
        assert len(indexes) == shape[0], (len(indexes), shape[0])
        self.array = array
        self.dtype = array.dtype
        self.cache = np.full(shape, np.nan, dtype=self.dtype)
        self.indexes = indexes

    def __setitem__(self, key, value):
        self.cache[key] = value

    def flush(self):
        for i in range(self.cache.shape[0]):
            global_i = self.indexes[i]
            self.array[global_i] = self.cache[i]


class ReindexFirst:
    def __init__(self, indexes):
        self.indexes = indexes

    def __call__(self, first, *others):
        if isinstance(first, int):
            return (self.indexes[first], *others)

        if isinstance(first, slice):
            start, stop, step = first.start, first.stop, first.step
            start = self.indexes[start]
            stop = self.indexes[stop]
            return (slice(start, stop, step), *others)
        if isinstance(first, tuple):
            return ([self.indexes[_] for _ in first], *others)

        raise NotImplementedError(type(first))


class DataWriter:
    def __init__(self, parts, full_array, owner):
        self.full_array = full_array

        self.path = owner.path
        self.statistics_registry = owner.statistics_registry
        self.registry = owner.registry
        self.print = owner.print
        self.dates = owner.dates
        self.variables_names = owner.variables_names

        self.append_axis = owner.output.append_axis
        self.n_groups = len(owner.groups)

    def date_to_index(self, date):
        if isinstance(date, str):
            date = np.datetime64(date)
        if isinstance(date, datetime.datetime):
            date = np.datetime64(date)
        assert type(date) is type(self.dates[0]), (type(date), type(self.dates[0]))
        return np.where(self.dates == date)[0][0]

    def write(self, result, igroup, dates):
        cube = result.get_cube()
        assert cube.extended_user_shape[0] == len(dates), (cube.extended_user_shape[0], len(dates))
        dates_in_data = cube.user_coords["valid_datetime"]
        dates_in_data = [datetime.datetime.fromisoformat(_) for _ in dates_in_data]
        assert dates_in_data == list(dates), (dates_in_data, list(dates))

        assert isinstance(igroup, int), igroup

        shape = cube.extended_user_shape

        msg = f"Building data for group {igroup}/{self.n_groups} ({shape=} in {self.full_array.shape=})"
        LOG.info(msg)
        self.print(msg)

        indexes = [self.date_to_index(d) for d in dates_in_data]
        array = ViewCacheArray(self.full_array, shape=shape, indexes=indexes)
        self.load_datacube(cube, array)

        stats = compute_statistics(array.cache, self.variables_names)
        dates = cube.user_coords["valid_datetime"]
        self.statistics_registry.write(indexes, stats, dates=dates)

        array.flush()
        self.registry.set_flag(igroup)

    def load_datacube(self, cube, array):
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        self.print(f"Loading datacube: {cube}")
        bar = progress_bar(
            iterable=cube.iterate_cubelets(reading_chunks),
            total=total,
            desc=f"Loading datacube {cube}",
        )
        for i, cubelet in enumerate(bar):
            now = time.time()
            data = cubelet.to_numpy()
            local_indexes = cubelet.coords
            load += time.time() - now

            name = self.variables_names[local_indexes[1]]
            check_data_values(data[:], name=name, log=[i, data.shape, local_indexes])

            bar.set_description(f"Loading {i}/{total} {name} {str(cubelet)} ({data.shape})")

            now = time.time()
            array[local_indexes] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now
        LOG.info("Written.")
        msg = f"Elapsed: {seconds(time.time() - start)}, load time: {seconds(load)}, write time: {seconds(save)}."
        self.print(msg)
        LOG.info(msg)
