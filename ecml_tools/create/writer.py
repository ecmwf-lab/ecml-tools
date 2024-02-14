# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import time
import warnings

import numpy as np

from .check import check_data_values
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
                parts = [
                    x
                    for x in range(total)
                    if x >= (i_chunk - 1) * chunk_size and x < i_chunk * chunk_size
                ]

        parts = [int(_) for _ in parts]
        LOG.info(f"Running parts: {parts}")
        if not parts:
            warnings.warn(f"Nothing to do for chunk {i_chunk}/{n_chunks}.")

        self.parts = parts

    def __call__(self, i):
        if self.parts is None:
            return True
        return i in self.parts


class ArrayLike:
    def __init__(self, array, shape):
        self.array = array
        self.shape = shape

    def flush():
        pass

    def new_key(self, key, values_shape):
        return key


class FastWriteArray(ArrayLike):
    """
    A class that provides a caching mechanism for writing to a NumPy-like array.

    The `FastWriteArray` instance is initialized with a NumPy-like array and its shape.
    The array is used to store the final data, while the cache is used to temporarily
    store the data before flushing it to the array. The cache is a NumPy array of the same
    shape as the final array, initialized with zeros.

    The `flush` method copies the contents of the cache to the final array.
    """

    def __init__(self, array, shape):
        self.array = array
        self.shape = shape
        self.dtype = array.dtype
        self.cache = np.zeros(shape, dtype=self.dtype)

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __getitem__(self, key):
        return self.cache[key]

    def new_key(self, key, values_shape):
        return self.array.new_key(key, values_shape)

    def flush(self):
        self.array[:] = self.cache[:]

    def compute_statistics_and_key(self, variables_names):
        from .statistics import compute_statistics

        now = time.time()
        stats = compute_statistics(self.cache, variables_names)
        LOG.info(f"Computed statistics in {seconds(time.time()-now)}.")

        new_key = self.new_key(slice(None, None), self.shape)

        assert isinstance(self.array, OffsetView)
        assert self.array.axis == 0, self.array.axis
        new_key = (new_key[0], slice(None, None))

        return new_key, stats


class OffsetView(ArrayLike):
    """
    A view on a portion of the large_array.
    'axis' is the axis along which the offset applies.
    'shape' is the shape of the view.
    """

    def __init__(self, large_array, *, offset, axis, shape):
        self.large_array = large_array
        self.dtype = large_array.dtype
        self.offset = offset
        self.axis = axis
        self.shape = shape

    def new_key(self, key, values_shape):
        if isinstance(key, slice):
            # Ensure that the slice covers the entire view along the axis.
            assert key.start is None and key.stop is None, key

            # Create a new key for indexing the large array.
            new_key = tuple(
                (
                    slice(self.offset, self.offset + values_shape[i])
                    if i == self.axis
                    else slice(None)
                )
                for i in range(len(self.shape))
            )
        else:
            # For non-slice keys, adjust the key based on the offset and axis.
            new_key = tuple(
                k + self.offset if i == self.axis else k for i, k in enumerate(key)
            )
        return new_key

    def __setitem__(self, key, values):
        new_key = self.new_key(key, values.shape)

        start = time.time()
        LOG.info("Writing data to disk")
        self.large_array[new_key] = values
        LOG.info(f"Writing data done in {seconds(time.time()-start)}.")


class DataWriter:
    def __init__(self, parts, full_array, parent, print=print):
        self.parent = parent
        self.full_array = full_array

        self.path = parent.path
        self.statistics_registry = parent.statistics_registry
        self.registry = parent.registry
        self.print = parent.print

        self.append_axis = parent.output.append_axis
        self.n_cubes = parent.groups.n_groups

    def write(self, result, igroup,dates):
        cube = result.get_cube()
        assert cube.shape[0] == len(dates), (cube.shape[0], len(dates))
        assert cube.coords['dates'] == dates, (cube.coords['dates'], dates)
        self.write_cube(cube, igroup)

    @property
    def variables_names(self):
        return self.parent.variables_names

    def write_cube(self, cube, icube):
        assert isinstance(icube, int), icube

        shape = cube.extended_user_shape

        slice = self.registry.get_slice_for(icube)
        LOG.info(
            (
                f"Building dataset '{self.path}' i={icube} total={self.n_cubes} "
                f"(total shape ={shape}) at {slice}, {self.full_array.chunks=}"
            )
        )
        self.print(
            f"Building dataset (total shape ={shape}) at {slice}, {self.full_array.chunks=}"
        )

        offset = slice.start
        array = OffsetView(
            self.full_array, offset=offset, axis=self.append_axis, shape=shape
        )
        array = FastWriteArray(array, shape=shape)
        self.load_datacube(cube, array)

        new_key, stats = array.compute_statistics_and_key(self.variables_names)
        self.statistics_registry[new_key] = stats

        array.flush()

        self.registry.set_flag(icube)

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
            cubelet_coords = cubelet.coords

            bar.set_description(
                f"Loading {i}/{total} {str(cubelet)} ({data.shape}) {cube=}"
            )
            load += time.time() - now

            j = cubelet_coords[1]

            check_data_values(
                data[:],
                name=self.variables_names[j],
                log=[i, j, data.shape, cubelet_coords],
            )

            now = time.time()
            array[cubelet_coords] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now

        LOG.info("Written.")

        self.print(
            f"Elapsed: {seconds(time.time() - start)},"
            f" load time: {seconds(load)},"
            f" write time: {seconds(save)}."
        )
        LOG.info(
            f"Elapsed: {seconds(time.time() - start)},"
            f" load time: {seconds(load)},"
            f" write time: {seconds(save)}."
        )
