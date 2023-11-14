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
import os
import re
import time
import uuid
import warnings
from functools import cached_property
from contextlib import contextmanager

import numpy as np
import tqdm

from climetlab.core.order import build_remapping  # noqa:F401
from climetlab.utils import progress_bar
from .config import LoadersConfig
from .utils import _prepare_serialisation, normalize_and_check_dates
from climetlab.utils.humanize import bytes, seconds


LOG = logging.getLogger(__name__)

VERSION = "0.13"


class ArrayLike:
    def flush():
        pass


class DummyArrayLike(ArrayLike):
    """"""

    def __init__(self, array, shape):
        self.array = array

    def __getattribute__(self, __name: str):
        return super().__getattribute__(__name)

    def new_key(self, key, values_shape):
        return key


class FastWriter(ArrayLike):
    """
    A class that provides a caching mechanism for writing to a NumPy-like array.

    The `FastWriter` instance is initialized with a NumPy-like array and its shape.
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

    def compute_statistics(self, statistics_registry, names):
        nvars = self.shape[1]

        stats_shape = (self.shape[0], nvars)

        count = np.zeros(stats_shape, dtype=np.int64)
        sums = np.zeros(stats_shape, dtype=np.float64)
        squares = np.zeros(stats_shape, dtype=np.float64)

        minimum = np.zeros(stats_shape, dtype=np.float64)
        maximum = np.zeros(stats_shape, dtype=np.float64)

        for i, chunk in enumerate(self.cache):
            values = chunk.reshape((nvars, -1))
            minimum[i] = np.min(values, axis=1)
            maximum[i] = np.max(values, axis=1)
            sums[i] = np.sum(values, axis=1)
            squares[i] = np.sum(values * values, axis=1)
            count[i] = values.shape[1]

        stats = {
            "minimum": minimum,
            "maximum": maximum,
            "sums": sums,
            "squares": squares,
            "count": count,
        }
        new_key = self.array.new_key(slice(None, None), self.shape)
        assert self.array.axis == 0, self.array.axis
        # print("new_key", new_key, self.array.offset, self.array.axis)
        new_key = new_key[0]
        statistics_registry[new_key] = stats
        return stats

    def save_statistics(self, icube, statistics_registry, names):
        now = time.time()
        self.compute_statistics(statistics_registry, names)
        LOG.info(f"Computed statistics in {seconds(time.time()-now)}.")
        # for k, v in stats.items():
        #     with open(f"stats_{icube}_{k}.npy", "wb") as f:
        #         np.save(f, v)


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
            print(self.shape)
            assert key.start is None and key.stop is None, key

            # Create a new key for indexing the large array.
            new_key = tuple(
                slice(self.offset, self.offset + values_shape[i])
                if i == self.axis
                else slice(None)
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


class CubesFilter:
    def __init__(self, *, loader, parts, **kwargs):
        self.loader = loader

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

                total = len(self.loader.registry.get_flags())
                assert i_chunk > 0, f"Chunk number {i_chunk} must be positive."
                if n_chunks > total:
                    warnings.warn(
                        f"Number of chunks {n_chunks} is larger than the total number of chunks: {total}+1."
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
            warnings.warn(f"Nothing to do for chunk {i_chunk}.")

        self.parts = parts

    def __call__(self, i):
        if self.parts is None:
            return True
        return i in self.parts


class Loader:
    def __init__(self, *, path, config, print=print, partial=False, **kwargs):
        np.seterr(
            all="raise"
        )  # Catch all floating point errors, including overflow, sqrt(<0), etc

        self.main_config = LoadersConfig(config)
        self.input_handler = self.main_config.input_handler(partial)
        self.path = path
        self.kwargs = kwargs
        self.print = print

    @classmethod
    def from_config(cls, *, config, path, **kwargs):
        # config is the path to the config file
        # or a dict with the config
        return cls(config=config, path=path, **kwargs)

    @classmethod
    def cache_context(cls, cache_dir):
        @contextmanager
        def no_cache_context():
            yield

        if cache_dir is None:
            return no_cache_context()

        from climetlab import settings

        os.makedirs(cache_dir, exist_ok=True)
        return settings.temporary("cache-directory", cache_dir)

    @classmethod
    def already_exists(self, path):
        import zarr

        try:
            zarr.open(path, "r")
            return True
        except zarr.errors.PathNotFoundError:
            return False

    @cached_property
    def registry(self):
        raise NotImplementedError()

    @cached_property
    def statistics_registry(self):
        raise NotImplementedError()

    @classmethod
    def initialise_dataset(
        cls, path, config, force=False, no_check_name=True, cache_dir=None
    ):
        # check path
        _, ext = os.path.splitext(path)
        assert ext != "zarr", f"Unsupported extension={ext}"

        if cls.already_exists(path) and not force:
            raise Exception(f"{path} already exists. Use --force to overwrite.")

        with cls.cache_context(cache_dir):
            loader = cls.from_config(partial=True, path=path, config=config)
            loader.initialise(check_name=not no_check_name)

    def initialise(self, check_name=True):
        """Create empty dataset"""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        total_shape = self.input_handler.shape
        self.print(f"total_shape = {total_shape}")
        print("-------------------------")

        grid_points = self.input_handler.grid_points
        print(f"gridpoints size: {[len(i) for i in grid_points]}")
        print("-------------------------")

        dates = self.input_handler.get_datetimes()
        self.print(f"Found {len(dates)} datetimes.")
        print(
            f"Dates: Found {len(dates)} datetimes, in {self.input_handler.n_cubes} cubes: ",
            end="",
        )
        lengths = [str(len(c.get_datetimes())) for c in self.input_handler.iter_cubes()]
        print("+".join(lengths))
        self.print(f"Found {len(dates)} datetimes {'+'.join(lengths)}.")
        print("-------------------------")

        names = self.input_handler.variables
        self.print(f"Found {len(names)} variables : {','.join(names)}.")

        names_from_config = self.main_config.output.order_by[
            self.main_config.output.statistics
        ]
        assert (
            names == names_from_config
        ), f"Requested= {names_from_config} Actual= {names}"

        resolution = self.input_handler.resolution
        print(f"{resolution=}")

        chunks = self.input_handler.chunking
        print(f"{chunks=}")
        dtype = self.main_config.output.dtype

        self.print(
            f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}"
        )

        frequency = self.input_handler.frequency
        assert isinstance(frequency, int), frequency

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = _prepare_serialisation(self.main_config)

        metadata["description"] = self.main_config.description
        metadata["resolution"] = resolution

        metadata["data_request"] = self.input_handler.data_request

        metadata["order_by"] = self.main_config.output.order_by
        metadata["remapping"] = self.main_config.output.remapping
        metadata["flatten_grid"] = self.main_config.output.flatten_grid
        metadata["ensemble_dimension"] = self.main_config.output.ensemble_dimension

        metadata["variables"] = names
        metadata["version"] = VERSION
        metadata["frequency"] = frequency
        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()

        if check_name:
            basename, ext = os.path.splitext(os.path.basename(self.path))
            ds_name = DatasetName(
                basename,
                resolution,
                dates[0],
                dates[-1],
                frequency,
            )
            ds_name.raise_if_not_valid(print=self.print)

        for i, loop in enumerate(self.input_handler.loops):
            print(f"Loop {i}: ", loop._info)

        if len(dates) != total_shape[0]:
            raise ValueError(
                f"Final date size {len(dates)} (from {dates[0]} to {dates[-1]}, {frequency=}) "
                f"does not match data shape {total_shape[0]}. {total_shape=}"
            )

        dates = normalize_and_check_dates(
            dates,
            metadata["start_date"],
            metadata["end_date"],
            metadata["frequency"],
        )

        metadata.update(self.main_config.get("force_metadata", {}))

        ###############################################################
        # write data
        ###############################################################
        # this is backend specific
        self.initialise_dataset_backend()

        self.update_metadata(**metadata)

        self._add_dataset(name="data", chunks=chunks, dtype=dtype, shape=total_shape)
        self._add_dataset(name="dates", array=dates)
        self._add_dataset(name="latitudes", array=grid_points[0])
        self._add_dataset(name="longitudes", array=grid_points[1])

        self.registry.create(lengths=lengths)
        self.statistics_registry.create()

        self.registry.add_to_history("init finished")

    def initialise_dataset_backend():
        raise NotImplementedError()

    def iter_loops(self):
        for vars in self.input_handler.iter_loops():
            yield vars

    def _compute_lengths(self, multiply):
        def squeeze_dict(dic):
            keys = list(dic.keys())
            assert len(dic) == 1, keys
            return dic[keys[0]]

        lengths = []
        for i, vars in enumerate(self.iter_loops()):
            lst = squeeze_dict(vars)
            assert isinstance(lst, (tuple, list)), lst
            lengths.append(len(lst))
            print("i vars", i, vars, lengths, lst, f"{multiply=}")

        lengths = [x * multiply for x in lengths]
        return lengths

    @property
    def _variables_names(self):
        return self.main_config.output.order_by[self.main_config.output.statistics]

    def load(self, **kwargs):
        import zarr

        self.z = zarr.open(self.path, mode="r+")
        self.registry.add_to_history("loading_data_start", parts=kwargs.get("parts"))

        filter = CubesFilter(loader=self, **kwargs)
        ncubes = self.input_handler.n_cubes
        for icube, cubecreator in enumerate(self.input_handler.iter_cubes()):
            if not filter(icube):
                continue
            if self.registry.get_flag(icube):
                LOG.info(f" -> Skipping {icube} total={ncubes} (already done)")
                continue
            self.print(f" -> Processing i={icube} total={ncubes}")

            cube = cubecreator.to_cube()
            shape = cube.extended_user_shape
            chunks = cube.chunking(self.input_handler.output.chunking)
            axis = self.input_handler.output.append_axis

            slice = self.registry.get_slice_for(icube)

            LOG.info(
                f"Building ZARR '{self.path}' i={icube} total={ncubes} (total shape ={shape}) at {slice}, {chunks=}"
            )
            self.print(f"Building ZARR (total shape ={shape}) at {slice}, {chunks=}")

            offset = slice.start
            array = OffsetView(self.z["data"], offset=offset, axis=axis, shape=shape)
            array = FastWriter(array, shape=shape)
            self.load_datacube(cube, array)

            array.save_statistics(
                icube, self.statistics_registry, self._variables_names
            )

            array.flush()

            self.registry.set_flag(icube)

        self.registry.add_to_history("loading_data_end", parts=kwargs.get("parts"))
        self.registry.add_provenance(name="provenance_load")

    def load_datacube(self, cube, array):
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        self.print(f"Loading datacube {cube}")
        bar = progress_bar(
            iterable=cube.iterate_cubelets(reading_chunks),
            total=total,
            desc=f"Loading datacube {cube}",
        )
        for i, cubelet in enumerate(bar):
            now = time.time()
            data = cubelet.to_numpy()
            bar.set_description(
                f"Loading {i}/{total} {str(cubelet)} ({data.shape}) {cube=}"
            )
            load += time.time() - now

            j = cubelet.extended_icoords[1]
            check_data_values(
                data[:],
                name=self._variables_names[j],
                log=[i, j, data.shape, cubelet.extended_icoords],
            )

            now = time.time()
            array[cubelet.extended_icoords] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now

        LOG.info("Written")
        self.print_info()
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

    def statistics_start_indice(self):
        return self._statistics_subset_indices[0]

    def statistics_end_indice(self):
        return self._statistics_subset_indices[1]

    def _actual_statistics_start(self):
        return self._statistics_subset_indices[2]

    def _actual_statistics_end(self):
        return self._statistics_subset_indices[3]

    @cached_property
    def _statistics_subset_indices(self):
        statistics_start = self.main_config.output.get("statistics_start")
        statistics_end = self.main_config.output.get("statistics_end")
        try:
            from ecml_tools.data import open_dataset
        except ImportError:
            raise Exception("Need to pip install ecml_tools[zarr]")

        if statistics_end is None:
            warnings.warn(
                "No statistics_end specified, using last date of the dataset."
            )
        ds = open_dataset(self.path)
        subset = ds.dates_interval_to_indices(statistics_start, statistics_end)

        return (subset[0], subset[-1], ds.dates[subset[0]], ds.dates[subset[-1]])


class ZarrLoader(Loader):
    writer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = None

    @cached_property
    def registry(self):
        from .zarr import ZarrBuiltRegistry  # TODO

        return ZarrBuiltRegistry(self.path)

    @cached_property
    def statistics_registry(self):
        from .zarr import ZarrStatisticsRegistry  # TODO

        return ZarrStatisticsRegistry(self.path)

    @classmethod
    def from_dataset(cls, *, config, path, **kwargs):
        import zarr

        assert os.path.exists(path), path
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        # config = yaml.safe_load(z.attrs["_yaml_dump"])["_create_yaml_config"]
        kwargs.get("print", print)("Config loaded from zarr: ", config)
        return cls.from_config(config=config, path=path, **kwargs)

    def initialise_dataset_backend(self):
        import zarr

        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def update_metadata(self, **kwargs):
        import zarr

        z = zarr.open(self.path, mode="w+")
        for k, v in kwargs.items():
            if isinstance(v, np.datetime64):
                v = v.astype(datetime.datetime)
            if isinstance(v, datetime.date):
                v = v.isoformat()
            z.attrs[k] = v

    def _add_dataset(self, mode="r+", **kwargs):
        import zarr
        from .zarr import add_zarr_dataset

        z = zarr.open(self.path, mode=mode)
        return add_zarr_dataset(zarr_root=z, **kwargs)

    def print_info(self):
        assert self.z is not None
        try:
            print(self.z["data"].info)
        except Exception as e:
            print(e)
        print("...")
        try:
            print(self.z["data"].info)
        except Exception as e:
            print(e)

    def add_total_size(self, **kwargs):
        size, n = compute_directory_size(self.path)
        self.update_metadata(total_size=size, total_number_of_files=n)

    def add_statistics(self, no_write, **kwargs):
        do_write = not no_write

        incomplete = not all(self.registry.get_flags(sync=False))
        if do_write and incomplete:
            raise Exception(
                f"Zarr {self.path} is not fully built, not writing statistics."
            )

        statistics_start = self.main_config.output.get("statistics_start")
        statistics_end = self.main_config.output.get("statistics_end")

        if do_write:
            self.registry.add_to_history(
                "compute_statistics_start",
                start=statistics_start,
                end=statistics_end,
            )

        try:
            from ecml_tools.data import open_dataset
        except ImportError:
            raise Exception("Need to pip install ecml_tools")
        ds = open_dataset(self.path)

        stats = self.compute_statistics(ds, statistics_start, statistics_end)

        print(
            "\n".join(
                (
                    f"{v.rjust(10)}: "
                    f"min/max = {stats['minimum'][j]:.6g} {stats['maximum'][j]:.6g}"
                    "   \t:   "
                    f"mean/stdev = {stats['mean'][j]:.6g} {stats['stdev'][j]:.6g}"
                )
                for j, v in enumerate(ds.variables)
            )
        )

        if do_write:
            for k in [
                "mean",
                "stdev",
                "minimum",
                "maximum",
                "sums",
                "squares",
                "count",
            ]:
                self._add_dataset(name=k, array=stats[k])

            self.update_metadata(
                statistics_start_date=self._actual_statistics_start(),
                statistics_end_date=self._actual_statistics_end(),
            )

            self.registry.add_to_history(
                "compute_statistics_end",
                start=statistics_start,
                end=statistics_end,
            )

            self.registry.add_provenance(name="provenance_statistics")

    def compute_statistics(self, ds, statistics_start, statistics_end):
        i_start = self.statistics_start_indice()
        i_end = self.statistics_end_indice()

        i_len = i_end + 1 - i_start

        self.print(
            f"Statistics computed on {i_len}/{len(ds.dates)} samples "
            f"first={ds.dates[i_start]} "
            f"last={ds.dates[i_end]}"
        )
        if i_end < i_start:
            raise ValueError(
                f"Cannot compute statistics on an empty interval."
                f" Requested : {ds.dates[i_start]} {ds.dates[i_end]}."
                f" Available: {ds.dates[0]=} {ds.dates[-1]=}"
            )

        reg = self.statistics_registry

        maximum = reg.get_by_name("maximum")[i_start : i_end + 1]
        minimum = reg.get_by_name("minimum")[i_start : i_end + 1]
        sums = reg.get_by_name("sums")[i_start : i_end + 1]
        squares = reg.get_by_name("squares")[i_start : i_end + 1]
        count = reg.get_by_name("count")[i_start : i_end + 1]

        assert len(maximum) == i_len, (len(maximum), i_len)
        assert len(minimum) == i_len, (len(minimum), i_len)
        assert len(sums) == i_len, (len(sums), i_len)
        assert len(squares) == i_len, (len(squares), i_len)
        assert len(count) == i_len, (len(count), i_len)

        assert not np.isnan(minimum).any(), minimum
        assert not np.isnan(maximum).any(), maximum
        assert not np.isnan(sums).any(), sums
        assert not np.isnan(squares).any(), squares
        # assert all(count > 0), count

        _minimum = np.amin(minimum, axis=0)
        _maximum = np.amax(maximum, axis=0)
        _count = np.sum(count, axis=0)
        _sums = np.sum(sums, axis=0)
        _squares = np.sum(squares, axis=0)
        _mean = _sums / _count

        assert all(_count[0] == c for c in _count), _count

        x = _squares / _count - _mean * _mean
        # remove negative variance due to numerical errors
        # x[- 1e-15 < (x / (np.sqrt(_squares / _count) + np.abs(_mean))) < 0] = 0
        if not (x >= 0).all():
            print(x)
            print(ds.variables)
            print(_count)
            for i, (var, y) in enumerate(zip(ds.variables, x)):
                if y < 0:
                    print(
                        var,
                        y,
                        _maximum[i],
                        _minimum[i],
                        _mean[i],
                        _count[i],
                        _sums[i],
                        _squares[i],
                    )

                    print(var, np.min(sums[i]), np.max(sums[i]), np.argmin(sums[i]))
                    print(
                        var,
                        np.min(squares[i]),
                        np.max(squares[i]),
                        np.argmin(squares[i]),
                    )
                    print(var, np.min(count[i]), np.max(count[i]), np.argmin(count[i]))

            raise ValueError("Negative variance")

        _stdev = np.sqrt(x)

        stats = {
            "mean": _mean,
            "stdev": _stdev,
            "minimum": _minimum,
            "maximum": _maximum,
            "sums": _sums,
            "squares": _squares,
            "count": _count,
        }

        for v in stats.values():
            assert v.shape == stats["mean"].shape

        for i, name in enumerate(ds.variables):
            check_stats(**{k: v[i] for k, v in stats.items()}, msg=f"{i} {name}")

        return stats
