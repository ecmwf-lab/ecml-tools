# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import glob
import json
import logging
import os
import pickle
import shutil
import uuid
import warnings
from functools import cached_property

import numpy as np
import zarr

from ecml_tools.data import open_dataset
from ecml_tools.provenance import gather_provenance_info

from .check import DatasetName, check_data_values, check_stats
from .config import OutputSpecs, loader_config
from .input import FullLoops, PartialLoops
from .utils import bytes, compute_directory_sizes, normalize_and_check_dates
from .writer import DataWriter
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.13"


class Creator:
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

        if "statistics_tmp" in kwargs:
            statistics_tmp = kwargs["statistics_tmp"]
            if statistics_tmp is None:
                statistics_tmp = path + ".statistics"

            self.statistics_registry = StatisticsRegistry(
                statistics_tmp,
                history_callback=self.registry.add_to_history,
            )

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statistics_registry.delete()

        self.loops = PartialLoops(self.main_config, parent=self)
        self.output_specs = OutputSpecs(self.main_config, parent=self)

    def initialise(self, check_name=True):
        """Create empty dataset"""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        total_shape = self.output_specs.shape
        self.print(f"total_shape = {total_shape}")
        print("-------------------------")

        grid_points = self.loops.grid_points
        print(f"gridpoints size: {[len(i) for i in grid_points]}")
        print("-------------------------")

        dates = self.loops.get_datetimes()
        self.print(f"Found {len(dates)} datetimes.")
        print(
            f"Dates: Found {len(dates)} datetimes, in {self.loops.n_cubes} cubes: ",
            end="",
        )
        lengths = [str(len(c.get_datetimes())) for c in self.loops.iter_cubes()]
        print("+".join(lengths))
        self.print(f"Found {len(dates)} datetimes {'+'.join(lengths)}.")
        print("-------------------------")

        names = self.loops.variables
        self.print(f"Found {len(names)} variables : {','.join(names)}.")

        names_from_config = self.main_config.output.order_by[
            self.main_config.output.statistics
        ]
        assert (
            names == names_from_config
        ), f"Requested= {names_from_config} Actual= {names}"

        resolution = self.loops.resolution
        print(f"{resolution=}")

        chunks = self.output_specs.chunking
        print(f"{chunks=}")
        dtype = self.output_specs.config.dtype

        self.print(
            f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}"
        )

        frequency = self.loops.frequency
        assert isinstance(frequency, int), frequency

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

        metadata["description"] = self.main_config.description
        metadata["resolution"] = resolution

        metadata["data_request"] = self.loops.data_request

        metadata["order_by"] = self.output_specs.order_by
        metadata["remapping"] = self.output_specs.remapping
        metadata["flatten_grid"] = self.output_specs.flatten_grid
        metadata["ensemble_dimension"] = self.output_specs.config.ensemble_dimension

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loops = FullLoops(self.main_config, parent=self)
        self.output_specs = OutputSpecs(self.main_config, parent=self)

    def load(self, parts):
        self.registry.add_to_history("loading_data_start", parts=parts)
        data_writer = DataWriter(parts, parent=self)
        data_writer.write()
        self.registry.add_to_history("loading_data_end", parts=parts)
        self.registry.add_provenance(name="provenance_load")


class StatisticsCreator(Creator):
    def __init__(
        self,
        statistics_from_data=None,
        statistics_output=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.statistics_output = statistics_output
        self.statistics_from_data = statistics_from_data

        incomplete = not all(self.registry.get_flags(sync=False))
        if incomplete and not force:
            raise Exception(
                f"Zarr {self.path} is not fully built, not writting statistics."
            )

        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables

        statistics_start = self.main_config.output.get("statistics_start")
        statistics_end = self.main_config.output.get("statistics_end")

        subset = ds.dates_interval_to_indices(statistics_start, statistics_end)
        self.i_start = subset[0]
        self.i_end = subset[-1]
        self.date_start = ds.dates[subset[0]]
        self.date_end = ds.dates[subset[-1]]

        i_len = self.i_end + 1 - self.i_start
        self.print(f"Statistics computed on {i_len}/{len(ds.dates)} samples ")
        print(f"Requested ({i_len}): from {self.date_start} to {self.date_end}.")
        print(f"Available ({len(ds.dates)}): from {ds.dates[0]} to {ds.dates[-1]}.")
        if i_len < 1:
            raise ValueError(f"Cannot compute statistics on an empty interval.")

    def statistics(self):
        detailed_stats =self.get_detailed_stats()

        if self.statistics_output:
            print("Not writing detailed statistics because statistics_output is set")
        else:
            self.write_detailed_statistics(detailed_stats)

        detailed_stats = {
            k: v[self.i_start : self.i_end + 1] for k, v in detailed_stats.items()
        }

        stats = compute_aggregated_statistics(detailed_stats, self.variables_names)

        self.write_aggregated_statistics(stats)

    def get_detailed_stats(self):
        if self.statistics_from_data:
            build_temporary_statistics()

        expected_shape = (self.dataset_shape[0], self.dataset_shape[1])
        return self.statistics_registry.as_detailed_stats(expected_shape)

    def write_detailed_statistics(self, detailed_stats):

        z = zarr.open(self.path)["_build"]

        for k, v in detailed_stats.items():
            if k == "variables_names":
                continue
            add_zarr_dataset(zarr_root=z, name=k, array=v)

    def write_aggregated_statistics(self, stats):
        if self.statistics_output == "-":
            print(stats)
            return

        if self.statistics_output:
            print(dict(stats))
            stats.save(self.statistics_output)
            print(f"Wrote {self.statistics_output}")
            return

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
            statistics_start_date=str(self.date_start),
            statistics_end_date=str(self.date_end),
        )

        self.registry.add_provenance(name="provenance_statistics")

        self.registry.add_to_history(
            "compute_statistics_end",
            start=str(self.date_start),
            end=str(self.date_end),
            i_start=self.i_start,
            i_end=self.i_end,
        )
        print(f"Wrote statistics in {self.path}")


def compute_aggregated_statistics(data, variables_names):
    i_len = None
    for name, array in data.items():
        if i_len is None:
            i_len = len(array)
        assert len(array) == i_len, (name, len(array), i_len)

    for name, array in data.items():
        if name == "count":
            continue
        assert not np.isnan(array).any(), (name, array)

    _minimum = np.amin(data["minimum"], axis=0)
    _maximum = np.amax(data["maximum"], axis=0)
    _count = np.sum(data["count"], axis=0)
    _sums = np.sum(data["sums"], axis=0)
    _squares = np.sum(data["squares"], axis=0)
    _mean = _sums / _count

    assert all(_count[0] == c for c in _count), _count

    x = _squares / _count - _mean * _mean
    # remove negative variance due to numerical errors
    # x[- 1e-15 < (x / (np.sqrt(_squares / _count) + np.abs(_mean))) < 0] = 0
    check_variance_is_positive(
        x, variables_names, _minimum, _maximum, _mean, _count, _sums, _squares
    )
    _stdev = np.sqrt(x)

    stats = Statistics(
        minimum=_minimum,
        maximum=_maximum,
        mean=_mean,
        count=_count,
        sums=_sums,
        squares=_squares,
        stdev=_stdev,
        variables_names=variables_names,
    )

    return stats


class Statistics(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check()

    @property
    def size(self):
        return len(self["variables_names"])

    def check(self):
        for k, v in self.items():
            if k == "variables_names":
                assert len(v) == self.size
                continue
            assert v.shape == (self.size,)
            if k == "count":
                assert (v >= 0).all(), (k, v)
                assert v.dtype == np.int64, (k, v)
                continue
            if k == "stdev":
                assert (v >= 0).all(), (k, v)
            assert v.dtype == np.float64, (k, v)

        for i, name in enumerate(self["variables_names"]):
            check_stats(**{k: v[i] for k, v in self.items()}, msg=f"{i} {name}")
            check_data_values(self["minimum"][i], name=name)

    def __str__(self):
        return "\n".join(
            (
                f"{self['variables_names'][j].rjust(10)}: "
                f"min/max = {self['minimum'][j]:.6g} {self['maximum'][j]:.6g}"
                "   \t:   "
                f"mean/stdev = {self['mean'][j]:.6g} {self['stdev'][j]:.6g}"
            )
            for j in range(self.size)
        )

    def save(self, filename):
        assert filename.endswith(".json"), filename
        dic = {k: v.tolist() for k, v in self.items()}
        with open(filename, "w") as f:
            json.dump(dic, f)

    def load(self, filename):
        assert filename.endswith(".json"), filename
        with open(filename, "r") as f:
            dic = json.load(f)

        dic_ = {}
        for k, v in dic.items():
            if k == "count":
                dic_[k] = np.array(v, dtype=np.int64)
                continue
            if k == "variables":
                dic_[k] = v
                continue
            dic_[k] = np.array(v, dtype=np.float64)
        return Statistics(dic_)


def check_variance_is_positive(
    x, variables_names, minimum, maximum, mean, count, sums, squares
):
    if (x >= 0).all():
        return
    print(x)
    print(variables_names)
    print(count)
    for i, (var, y) in enumerate(zip(variables_names, x)):
        if y >= 0:
            continue
        print(
            var,
            y,
            maximum[i],
            minimum[i],
            mean[i],
            count[i],
            sums[i],
            squares[i],
        )

        print(var, np.min(sums[i]), np.max(sums[i]), np.argmin(sums[i]))
        print(var, np.min(squares[i]), np.max(squares[i]), np.argmin(squares[i]))
        print(var, np.min(count[i]), np.max(count[i]), np.argmin(count[i]))

    raise ValueError("Negative variance")


class Registry:
    # names = [ "mean", "stdev", "minimum", "maximum", "sums", "squares", "count", ]
    # build_names = [ "minimum", "maximum", "sums", "squares", "count", ]

    def __init__(self, dirname, history_callback=None, overwrite=False):
        if history_callback is None:

            def dummy(*args, **kwargs):
                pass

            history_callback = dummy

        self.dirname = dirname
        self.overwrite = overwrite
        self.history_callback = history_callback

    def create(self):
        assert not os.path.exists(self.dirname), self.dirname
        os.makedirs(self.dirname, exist_ok=True)
        self.history_callback(
            f"{self.name}_registry_initialised", **{f"{self.name}_version": 2}
        )

        with open(os.path.join(self.dirname, 'proveance.json'), 'w') as f:
            json.dumps(gather_provenance_info(),f)

    def delete(self):
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def __setitem__(self, key, data):
        # if isinstance(key, slice):
        #     # this is just to make the filenames nicer.
        #     key_str = f"{key.start}_{key.stop}"
        #     if key.step is not None:
        #         key_str = f"{key_str}_{key.step}"
        # else:
        #     key_str = str(key_str)

        key_str = (
            str(key)
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace(",", "_")
            .replace("None", "x")
            .replace("__", "_")
            .lower()
        )
        path = os.path.join(self.dirname, f"{key_str}.npz")

        if not self.overwrite:
            assert not os.path.exists(path), f"{path} already exists"

        with open(path + '.tmp', "wb") as f:
            pickle.dump((key, data), f)
        shutil.move(path + '.tmp', path)
        LOG.info(f"Written {self.name} data for {key} in  {path}")

    def __iter__(self):
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.npz")

        LOG.info(
            f"Reading {self.name} data, found {len(files)} for {self.name} in  {self.dirname}"
        )

        assert len(files) > 0, f"No files found in {self.dirname}"

        key_strs = dict()
        for f in files:
            with open(f, "rb") as f:
                key, data = pickle.load(f)

            key_str = str(key)
            if key_str in key_strs:
                raise Exception(
                    f"Duplicate key {key}, found in {f} and {key_strs[key_str]}"
                )
            key_strs[key_str] = f

            yield key, data


class StatisticsRegistry(Registry):
    name = "statistics"

    def as_detailed_stats(self, shape):
        detailed_stats = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
        )

        flags = np.full(shape, False, dtype=np.bool)
        for key, data in self:
            assert isinstance(data, dict), data
            assert not np.any(flags[key]), f"Overlapping values for {key} {flags}"
            flags[key] = True
            for name, array in detailed_stats.items():
                d = data[name]
                array[key] = d
        assert np.all(flags), f"Missing statistics data for {np.where(flags==False)}"

        return detailed_stats


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
