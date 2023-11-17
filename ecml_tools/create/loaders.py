# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
import uuid
import warnings
from functools import cached_property

import numpy as np

from .check import DatasetName, check_stats
from .config import loader_config
from .utils import bytes, compute_directory_sizes, normalize_and_check_dates
from .writer import DataWriter
from .input import InputHandler

LOG = logging.getLogger(__name__)

VERSION = "0.13"


class Creator:
    partial = False

    def __init__(self, *, path, config, print=print, **kwargs):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise")

        # config is the path to the config file or a dict with the config
        assert isinstance(config, dict) or isinstance(config, str), config
        assert isinstance(path, str), path

        self.main_config = loader_config(config)

        self.path = path
        self.kwargs = kwargs
        self.print = print

        self.statistics_path = kwargs.get("statistics_path", path + ".statistics")

    @classmethod
    def from_config(cls, *, config, path, print=print, **kwargs):
        return cls(config=config, path=path, print=print, **kwargs)

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
        from .zarr import ZarrBuiltRegistry  # TODO

        return ZarrBuiltRegistry(self.path)

    @cached_property
    def statistics_registry(self):
        from .zarr import StatisticsRegistry  # TODO

        return StatisticsRegistry(
            self.statistics_path,
            history_callback=self.registry.add_to_history,
        )

    @classmethod
    def from_dataset(cls, *, path, print=print, **kwargs):
        import zarr

        assert os.path.exists(path), path
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        print("Config loaded from zarr: ", config)
        return cls.from_config(config=config, path=path, print=print, **kwargs)

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


class InitialiseCreator(Creator):
    partial = True

    def __init__(self, *, path, config, print=print, **kwargs):
        super().__init__(path=path, config=config, print=print, **kwargs)
        self.statistics_registry.delete()
        self.data_descriptor = InputHandler(self.main_config, partial=self.partial)
        self.input_handler = self.data_descriptor

    def initialise(self, check_name=True):
        """Create empty dataset"""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        total_shape = self.data_descriptor.shape
        self.print(f"total_shape = {total_shape}")
        print("-------------------------")

        grid_points = self.data_descriptor.grid_points
        print(f"gridpoints size: {[len(i) for i in grid_points]}")
        print("-------------------------")

        dates = self.data_descriptor.get_datetimes()
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

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

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

        self.initialise_dataset_backend()  # this is backend specific

        self.update_metadata(**metadata)

        self._add_dataset(name="data", chunks=chunks, dtype=dtype, shape=total_shape)
        self._add_dataset(name="dates", array=dates)
        self._add_dataset(name="latitudes", array=grid_points[0])
        self._add_dataset(name="longitudes", array=grid_points[1])

        self.registry.create(lengths=lengths)
        self.statistics_registry.create()

        self.registry.add_to_history("init finished")


class LoadCreator(Creator):
    def __init__(self, *, path, config, print=print, **kwargs):
        super().__init__(path=path, config=config, print=print, **kwargs)
        self.data_descriptor = InputHandler(self.main_config, partial=self.partial)
        self.cubes_provider = self.data_descriptor

    def load(self, parts):
        self.registry.add_to_history("loading_data_start", parts=parts)
        data_writer = DataWriter(parts, parent=self)
        data_writer.write()
        self.registry.add_to_history("loading_data_end", parts=parts)
        self.registry.add_provenance(name="provenance_load")


class StatisticsCreator(Creator):
    def __init__(self, *, path, config, print=print, **kwargs):
        super().__init__(path=path, config=config, print=print, **kwargs)
        self.input_handler = InputHandler(self.main_config, partial=self.partial)

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

    def add_statistics(self, no_write, **kwargs):
        do_write = not no_write

        incomplete = not all(self.registry.get_flags(sync=False))
        if do_write and incomplete:
            raise Exception(
                f"Zarr {self.path} is not fully built, not computing statistics."
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
        def check_variance_is_positive(x):
            if (x >= 0).all():
                return
            print(x)
            print(ds.variables)
            print(_count)
            for i, (var, y) in enumerate(zip(ds.variables, x)):
                if y >= 0:
                    continue
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
                    var, np.min(squares[i]), np.max(squares[i]), np.argmin(squares[i])
                )
                print(var, np.min(count[i]), np.max(count[i]), np.argmin(count[i]))

            raise ValueError("Negative variance")

        check_variance_is_positive(x)

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


class SizeCreator(Creator):
    def add_total_size(self):
        dic = compute_directory_sizes(self.path)

        size = dic["total_size"]
        n = dic["total_number_of_files"]

        print(f"Total size: {bytes(size)}")
        print(f"Total number of files: {n}")

        try:
            self.update_metadata(total_size=size, total_number_of_files=n)
        except PermissionError as e:
            print(f"Cannot update metadata ({e})")
