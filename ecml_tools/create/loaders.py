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
from functools import cached_property

import numpy as np
import zarr

from ecml_tools.data import open_dataset

from .check import DatasetName
from .config import OutputSpecs, loader_config
from .input import FullLoops, PartialLoops
from .statistics import StatisticsRegistry, compute_aggregated_statistics
from .utils import bytes, compute_directory_sizes, normalize_and_check_dates
from .writer import DataWriter
from .zarr import add_zarr_dataset
from .utils import progress_bar

LOG = logging.getLogger(__name__)

VERSION = "0.13"


class Loader:
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

        assert os.path.exists(path), f"Path {path} does not exist."
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


class InitialiseLoader(Loader):
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
        self.statistics_registry.create(exist_ok=False)
        self.add_to_history(
            f"statistics_registry_initialised", version=self.statistics_registry.version
        )

        self.registry.add_to_history("init finished")


class ContentLoader(Loader):
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


class StatisticsLoader(Loader):
    def __init__(
        self,
        statistics_from_data=None,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._write_to_dataset = True

        if statistics_from_data:
            print(
                "Recomputing temporary statistics, not writting statistics into dataset."
            )
            self._write_to_dataset = False
        if statistics_output:
            self._write_to_dataset = False

        if not statistics_from_data and (statistics_start or statistics_end):
            raise NotImplementedError("Not implemented yet.")

        self.statistics_output = statistics_output
        self.statistics_from_data = statistics_from_data

        incomplete = not all(self.registry.get_flags(sync=False))
        if incomplete and not force:
            self._write_to_dataset = False
            print(
                f"Zarr {self.path} is not fully built, not writting statistics into dataset."
            )

        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables

        statistics_start = statistics_start or self.main_config.output.get(
            "statistics_start"
        )
        statistics_end = statistics_end or self.main_config.output.get("statistics_end")

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
            raise ValueError("Cannot compute statistics on an empty interval.")

    def statistics(self):
        detailed_stats = self.get_detailed_stats()

        self.write_detailed_statistics(detailed_stats)

        detailed_stats = {
            k: v[self.i_start : self.i_end + 1] for k, v in detailed_stats.items()
        }

        stats = compute_aggregated_statistics(detailed_stats, self.variables_names)

        self.write_aggregated_statistics(stats)

    def get_detailed_stats(self):
        if self.statistics_from_data:
            self.build_temporary_statistics(self.path, self.statistics_registry)

        expected_shape = (self.dataset_shape[0], self.dataset_shape[1])
        return self.statistics_registry.as_detailed_stats(expected_shape)

    def build_temporary_statistics(self, path, statistics_registry):
        statistics_registry.create(exist_ok=True)
        from .statistics import compute_statistics

        self.print(
            (
                f"Building temporary statistics from data {self.path}. "
                f"From {self.date_start} to {self.date_end}"
            )
        )
        ds = open_dataset(self.path)

        shape = (self.i_end + 1 - self.i_start, len(ds.variables))
        detailed_stats = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
        )

        for i in progress_bar(
            desc="Computing Statistics", iterable=range(self.i_start, self.i_end + 1)
        ):
            data = ds[slice(i, i + 1), :]
            one = compute_statistics(data)
            for k, v in one.items():
                detailed_stats[k][i] = v

        key = (slice(self.i_start, self.i_end + 1), slice(None, None))
        print(f"✅ Saving statistics for {key} shape={detailed_stats['count'].shape}")
        statistics_registry[key] = detailed_stats
        exit()

    def write_detailed_statistics(self, detailed_stats):
        if not self._write_to_dataset:
            print("Not writing detailed statistics to zarr.")
            return

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

        if not self._write_to_dataset:
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


class SizeLoader(Loader):
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
