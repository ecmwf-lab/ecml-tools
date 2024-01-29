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
from .config import build_output, loader_config
from .group import build_groups
from .input import build_input
from .statistics import (
    StatisticsRegistry,
    compute_aggregated_statistics,
    compute_statistics,
)
from .utils import (
    bytes,
    compute_directory_sizes,
    normalize_and_check_dates,
    progress_bar,
    to_datetime,
)
from .writer import CubesFilter, DataWriter
from .zarr import ZarrBuiltRegistry, add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


class Loader:
    def __init__(self, *, path, print=print, **kwargs):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise")

        assert isinstance(path, str), path

        self.path = path
        self.kwargs = kwargs
        self.print = print

        statistics_tmp = kwargs.get("statistics_tmp") or self.path + ".statistics"

        self.statistics_registry = StatisticsRegistry(
            statistics_tmp,
            history_callback=self.registry.add_to_history,
        )

    @classmethod
    def from_config(cls, *, config, path, print=print, **kwargs):
        # config is the path to the config file or a dict with the config
        assert isinstance(config, dict) or isinstance(config, str), config
        return cls(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset_config(cls, *, path, print=print, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        print("Config loaded from zarr config: ", config)
        return cls.from_config(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset(cls, *, path, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        return cls(path=path, **kwargs)

    @property
    def order_by(self):
        return self.output.order_by

    @property
    def flatten_grid(self):
        return self.output.flatten_grid

    def read_dataset_metadata(self):
        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables

        z = zarr.open(self.path, "r")
        start = z.attrs.get("statistics_start_date")
        end = z.attrs.get("statistics_end_date")
        if start:
            start = to_datetime(start)
        if end:
            end = to_datetime(end)
        self._statistics_start_date_from_dataset = start
        self._statistics_end_date_from_dataset = end

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path)

    def initialise_dataset_backend(self):
        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def update_metadata(self, **kwargs):
        z = zarr.open(self.path, mode="w+")
        for k, v in kwargs.items():
            if isinstance(v, np.datetime64):
                v = v.astype(datetime.datetime)
            if isinstance(v, datetime.date):
                v = v.isoformat()
            z.attrs[k] = v

    def _add_dataset(self, mode="r+", **kwargs):
        z = zarr.open(self.path, mode=mode)
        return add_zarr_dataset(zarr_root=z, **kwargs)

    def get_zarr_chunks(self):
        z = zarr.open(self.path, mode="r")
        return z["data"].chunks

    def print_info(self):
        z = zarr.open(self.path, mode="r")
        try:
            print(z["data"].info)
        except Exception as e:
            print(e)


class InitialiseLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.statistics_registry.delete()

        self.groups = build_groups(*self.main_config.loop)
        self.output = build_output(self.main_config.output, parent=self)
        self.input = build_input(self.main_config.input, self)

        print(self.input)
        self.inputs = self.input.select(dates=None)
        all_dates = self.inputs.dates
        self.minimal_input = self.input.select(dates=[all_dates[0]])

        print("✅ GROUPS")
        print(self.groups)
        print("✅ ACTIONS")
        print(self.input._action)
        print("✅ ALL INPUTS")
        print(self.inputs)
        print("✅ MINIMAL INPUT")
        print(self.minimal_input)

    def initialise(self, check_name=True):
        """Create empty dataset"""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        dates = self.inputs.dates
        if self.groups.frequency != self.inputs.frequency:
            raise ValueError(
                f"Frequency mismatch: {self.groups.frequency} != {self.inputs.frequency}"
            )
        if self.groups.values[0] != self.inputs.dates[0]:
            raise ValueError(
                f"First date mismatch: {self.groups.values[0]} != {self.inputs.dates[0]}"
            )
        print("-------------------------")

        frequency = self.inputs.frequency
        assert isinstance(frequency, int), frequency

        self.print(f"Found {len(dates)} datetimes.")
        print(
            f"Dates: Found {len(dates)} datetimes, in {self.groups.n_groups} groups: ",
            end="",
        )
        lengths = [len(g) for g in self.groups.groups]
        self.print(
            f"Found {len(dates)} datetimes {'+'.join([str(_) for _ in lengths])}."
        )
        print("-------------------------")

        variables = self.minimal_input.variables
        self.print(f"Found {len(variables)} variables : {','.join(variables)}.")

        ensembles = self.minimal_input.ensembles
        self.print(
            f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}."
        )

        grid_points = self.minimal_input.grid_points
        print(f"gridpoints size: {[len(i) for i in grid_points]}")
        print("-------------------------")

        resolution = self.minimal_input.resolution
        print(f"{resolution=}")

        print("-------------------------")
        coords = self.minimal_input.coords
        coords["dates"] = dates
        total_shape = self.minimal_input.shape
        total_shape[0] = len(dates)
        self.print(f"total_shape = {total_shape}")
        print("-------------------------")

        chunks = self.output.get_chunking(coords)
        print(f"{chunks=}")
        dtype = self.output.dtype

        self.print(
            f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}"
        )

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

        metadata["description"] = self.main_config.description
        metadata["version"] = VERSION

        metadata["data_request"] = self.minimal_input.data_request
        metadata["remapping"] = self.output.remapping

        metadata["order_by"] = self.output.order_by_as_list
        metadata["flatten_grid"] = self.output.flatten_grid

        metadata["ensemble_dimension"] = len(ensembles)
        metadata["variables"] = variables
        metadata["resolution"] = resolution

        metadata["licence"] = self.main_config["licence"]
        metadata["copyright"] = self.main_config["copyright"]

        metadata["frequency"] = frequency
        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()

        # metadata["statistics_start_date"]=self.output.get("statistics_start")
        # metadata["statistics_end_date"]=self.output.get("statistics_end")

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

        self.initialise_dataset_backend()

        self.update_metadata(**metadata)

        self._add_dataset(name="data", chunks=chunks, dtype=dtype, shape=total_shape)
        self._add_dataset(name="dates", array=dates)
        self._add_dataset(name="latitudes", array=grid_points[0])
        self._add_dataset(name="longitudes", array=grid_points[1])

        self.registry.create(lengths=lengths)
        self.statistics_registry.create(exist_ok=False)
        self.registry.add_to_history(
            "statistics_registry_initialised", version=self.statistics_registry.version
        )

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())


class ContentLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.groups = build_groups(*self.main_config.loop)
        self.output = build_output(self.main_config.output, parent=self)
        self.input = build_input(self.main_config.input, self)

        self.read_dataset_metadata()

    def load(self, parts):
        self.registry.add_to_history("loading_data_start", parts=parts)

        z = zarr.open(self.path, mode="r+")
        data_writer = DataWriter(
            parts, parent=self, full_array=z["data"], print=self.print
        )

        total = len(self.registry.get_flags())
        n_groups = len(self.groups.groups)
        filter = CubesFilter(parts=parts, total=total)
        for igroup, group in enumerate(self.groups.groups):
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={n_groups} (already done)")
                continue
            if not filter(igroup):
                continue
            self.print(f" -> Processing {igroup} total={n_groups}")
            assert isinstance(group[0], datetime.datetime), group

            inputs = self.input.select(dates=group)
            data_writer.write(inputs, igroup)

        self.registry.add_to_history("loading_data_end", parts=parts)
        self.registry.add_provenance(name="provenance_load")
        self.statistics_registry.add_provenance(
            name="provenance_load", config=self.main_config
        )

        self.print_info()


class StatisticsLoader(Loader):
    main_config = {}

    def __init__(
        self,
        config=None,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
        force=False,
        recompute=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recompute = recompute

        self._write_to_dataset = True

        self.statistics_output = statistics_output
        if self.statistics_output:
            self._write_to_dataset = False

        if config:
            self.main_config = loader_config(config)

        self._statistics_start = statistics_start
        self._statistics_end = statistics_end

        self.check_complete(force=force)

        self.read_dataset_metadata()
        self.read_dataset_dates_metadata()

    def run(self):
        # if requested, recompute statistics from data
        # into the temporary statistics directory
        # (this should have been done already when creating the dataset content)
        if self.recompute:
            self.recompute_temporary_statistics()

        # compute the detailed statistics from temporary statistics directory
        detailed = self.get_detailed_stats()

        if self._write_to_dataset:
            self.write_detailed_statistics(detailed)

        # compute the aggregated statistics from the detailed statistics
        # for the selected dates
        selected = {k: v[self.i_start : self.i_end + 1] for k, v in detailed.items()}
        stats = compute_aggregated_statistics(selected, self.variables_names)

        if self._write_to_dataset:
            self.write_aggregated_statistics(stats)

    def check_complete(self, force):
        if self._complete:
            return
        if not force:
            raise Exception(
                f"❗Zarr {self.path} is not fully built. Use 'force' option."
            )
        if self._write_to_dataset:
            print(
                f"❗Zarr {self.path} is not fully built, not writting statistics into dataset."
            )
            self._write_to_dataset = False

    @property
    def statistics_start(self):
        user = self._statistics_start
        config = self.main_config.get("output", {}).get("statistics_start")
        dataset = self._statistics_start_date_from_dataset
        return user or config or dataset

    @property
    def statistics_end(self):
        user = self._statistics_end
        config = self.main_config.get("output", {}).get("statistics_end")
        dataset = self._statistics_end_date_from_dataset
        return user or config or dataset

    @property
    def _complete(self):
        return all(self.registry.get_flags(sync=False))

    def read_dataset_dates_metadata(self):
        ds = open_dataset(self.path)
        subset = ds.dates_interval_to_indices(
            self.statistics_start, self.statistics_end
        )
        self.i_start = subset[0]
        self.i_end = subset[-1]
        self.date_start = ds.dates[subset[0]]
        self.date_end = ds.dates[subset[-1]]

        # do not write statistics to dataset if dates do not match the ones in the dataset metadata
        start = self._statistics_start_date_from_dataset
        end = self._statistics_end_date_from_dataset

        start_ok = start is None or to_datetime(self.date_start) == start
        end_ok = end is None or to_datetime(self.date_end) == end
        if not (start_ok and end_ok):
            print(
                f"Statistics start/end dates {self.date_start}/{self.date_end} "
                f"do not match dates in the dataset metadata {start}/{end}. "
                f"Will not write statistics to dataset."
            )
            self._write_to_dataset = False

        def check():
            i_len = self.i_end + 1 - self.i_start
            self.print(f"Statistics computed on {i_len}/{len(ds.dates)} samples ")
            print(f"Requested ({i_len}): from {self.date_start} to {self.date_end}.")
            print(f"Available ({len(ds.dates)}): from {ds.dates[0]} to {ds.dates[-1]}.")
            if i_len < 1:
                raise ValueError("Cannot compute statistics on an empty interval.")

        check()

    def recompute_temporary_statistics(self):
        self.statistics_registry.create(exist_ok=True)

        self.print(
            (
                f"Building temporary statistics from data {self.path}. "
                f"From {self.date_start} to {self.date_end}"
            )
        )

        shape = (self.i_end + 1 - self.i_start, len(self.variables_names))
        detailed_stats = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
        )

        ds = open_dataset(self.path)
        key = (slice(self.i_start, self.i_end + 1), slice(None, None))
        for i in progress_bar(
            desc="Computing Statistics",
            iterable=range(self.i_start, self.i_end + 1),
        ):
            i_ = i - self.i_start
            data = ds[slice(i, i + 1), :]
            one = compute_statistics(data, self.variables_names)
            for k, v in one.items():
                detailed_stats[k][i_] = v

        print(f"✅ Saving statistics for {key} shape={detailed_stats['count'].shape}")
        self.statistics_registry[key] = detailed_stats
        self.statistics_registry.add_provenance(
            name="provenance_recompute_statistics", config=self.main_config
        )

    def get_detailed_stats(self):
        expected_shape = (self.dataset_shape[0], self.dataset_shape[1])
        try:
            return self.statistics_registry.as_detailed_stats(expected_shape)
        except self.statistics_registry.MissingDataException as e:
            missing_index = e.args[1]
            dates = open_dataset(self.path).dates
            missing_dates = dates[missing_index[0]]
            print(
                (
                    f"Missing dates: "
                    f"{missing_dates[0]} ... {missing_dates[len(missing_dates)-1]} "
                    f"({missing_dates.shape[0]} missing)"
                )
            )
            raise

    def write_detailed_statistics(self, detailed_stats):
        z = zarr.open(self.path)["_build"]
        for k, v in detailed_stats.items():
            if k == "variables_names":
                continue
            add_zarr_dataset(zarr_root=z, name=k, array=v)
        print("Wrote detailed statistics to zarr.")

    def write_aggregated_statistics(self, stats):
        if self.statistics_output == "-":
            print(stats)
            return

        if self.statistics_output:
            stats.save(self.statistics_output, provenance=dict(config=self.main_config))
            print(f"✅ Statistics written in {self.statistics_output}")
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

        self.registry.add_to_history(
            "compute_statistics_end",
            start=str(self.date_start),
            end=str(self.date_end),
            i_start=self.i_start,
            i_end=self.i_end,
        )
        print(f"Wrote statistics in {self.path}")


class SizeLoader(Loader):
    def __init__(self, path, print):
        self.path = path
        self.print = print

    def add_total_size(self):
        dic = compute_directory_sizes(self.path)

        size = dic["total_size"]
        n = dic["total_number_of_files"]

        print(f"Total size: {bytes(size)}")
        print(f"Total number of files: {n}")

        self.update_metadata(total_size=size, total_number_of_files=n)
