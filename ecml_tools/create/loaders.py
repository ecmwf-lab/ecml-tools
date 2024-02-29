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
from ecml_tools.utils.dates.groups import Groups

from .check import DatasetName
from .config import build_output, loader_config
from .input import build_input
from .statistics import TempStatistics
from .utils import bytes, compute_directory_sizes, normalize_and_check_dates
from .writer import CubesFilter, DataWriter
from .zarr import ZarrBuiltRegistry, add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


class Loader:
    def __init__(self, *, path, print=print, **kwargs):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        assert isinstance(path, str), path

        self.path = path
        self.kwargs = kwargs
        self.print = print

        statistics_tmp = kwargs.get("statistics_tmp") or self.path + ".statistics"

        self.statistics_registry = TempStatistics(statistics_tmp)

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

    def build_input(self):
        from climetlab.core.order import build_remapping

        builder = build_input(
            self.main_config.input,
            include=self.main_config.get("include", {}),
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
        )
        print("✅ INPUT_BUILDER")
        print(builder)
        return builder

    def build_statistics_dates(self, start, end):
        ds = open_dataset(self.path)
        subset = ds.dates_interval_to_indices(start, end)
        start, end = ds.dates[subset[0]], ds.dates[subset[-1]]
        return (
            start.astype(datetime.datetime).isoformat(),
            end.astype(datetime.datetime).isoformat(),
        )

    def read_dataset_metadata(self):
        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        z = zarr.open(self.path, "r")
        self.missing_dates = z.attrs.get("missing_dates", [])
        self.missing_dates = [np.datetime64(d) for d in self.missing_dates]

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path)

    def initialise_dataset_backend(self):
        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def update_metadata(self, **kwargs):
        print("Updating metadata", kwargs)
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

        print(self.main_config.dates)
        self.groups = Groups(**self.main_config.dates)
        print("✅ GROUPS")

        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()

        print(self.input)
        all_dates = self.groups.dates
        self.minimal_input = self.input.select([all_dates[0]])

        print("✅ GROUPS")
        print(self.groups)
        print("✅ MINIMAL INPUT")
        print(self.minimal_input)

    def initialise(self, check_name=True):
        """Create empty dataset."""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        dates = self.groups.dates
        print("-------------------------")

        frequency = dates.frequency
        assert isinstance(frequency, int), frequency

        self.print(f"Found {len(dates)} datetimes.")
        print(
            f"Dates: Found {len(dates)} datetimes, in {len(self.groups)} groups: ",
            end="",
        )
        print(f"Missing dates: {len(dates.missing)}")
        lengths = [len(g) for g in self.groups]
        self.print(f"Found {len(dates)} datetimes {'+'.join([str(_) for _ in lengths])}.")
        print("-------------------------")

        variables = self.minimal_input.variables
        self.print(f"Found {len(variables)} variables : {','.join(variables)}.")

        ensembles = self.minimal_input.ensembles
        self.print(f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}.")

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

        self.print(f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}")

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
        metadata["missing_dates"] = [_.isoformat() for _ in dates.missing]

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
        self.registry.add_to_history("statistics_registry_initialised", version=self.statistics_registry.version)

        statistics_start, statistics_end = self.build_statistics_dates(
            self.main_config.output.get("statistics_start"),
            self.main_config.output.get("statistics_end"),
        )
        self.update_metadata(
            statistics_start_date=statistics_start,
            statistics_end_date=statistics_end,
        )
        print(f"Will compute statistics from {statistics_start} to {statistics_end}")

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())


class ContentLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.groups = Groups(**self.main_config.dates)
        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()
        self.read_dataset_metadata()

    def load(self, parts):
        self.registry.add_to_history("loading_data_start", parts=parts)

        z = zarr.open(self.path, mode="r+")
        data_writer = DataWriter(parts, full_array=z["data"], owner=self)

        total = len(self.registry.get_flags())
        filter = CubesFilter(parts=parts, total=total)
        for igroup, group in enumerate(self.groups):
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue
            if not filter(igroup):
                continue
            self.print(f" -> Processing {igroup} total={len(self.groups)}")
            print("========", group)
            assert isinstance(group[0], datetime.datetime), group

            result = self.input.select(dates=group)
            data_writer.write(result, igroup, group)

        self.registry.add_to_history("loading_data_end", parts=parts)
        self.registry.add_provenance(name="provenance_load")
        self.statistics_registry.add_provenance(name="provenance_load", config=self.main_config)

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_statistics_start = statistics_start
        self.user_statistics_end = statistics_end

        self.statistics_output = statistics_output

        self.output_writer = {
            None: self.write_stats_to_dataset,
            "-": self.write_stats_to_stdout,
        }.get(self.statistics_output, self.write_stats_to_file)

        if config:
            self.main_config = loader_config(config)

        self.read_dataset_metadata()

    def _get_statistics_dates(self):
        dates = self.dates
        dtype = type(dates[0])

        # remove missing dates
        if self.missing_dates:
            assert type(self.missing_dates[0]) is dtype, (type(self.missing_dates[0]), dtype)
        dates = [d for d in dates if d not in self.missing_dates]

        # filter dates according the the start and end dates in the metadata
        z = zarr.open(self.path, mode="r")
        start, end = z.attrs.get("statistics_start_date"), z.attrs.get("statistics_end_date")
        start, end = np.datetime64(start), np.datetime64(end)
        assert type(start) is dtype, (type(start), dtype)
        dates = [d for d in dates if d >= start and d <= end]

        # filter dates according the the user specified start and end dates
        if self.user_statistics_start or self.user_statistics_end:
            start, end = self.build_statistics_dates(self.user_statistics_start, self.user_statistics_end)
            start, end = np.datetime64(start), np.datetime64(end)
            assert type(start) is dtype, (type(start), dtype)
            dates = [d for d in dates if d >= start and d <= end]

        return dates

    def run(self):
        dates = self._get_statistics_dates()
        stats = self.statistics_registry.get_aggregated(dates, self.variables_names)
        self.output_writer(stats)

    def write_stats_to_file(self, stats):
        stats.save(self.statistics_output, provenance=dict(config=self.main_config))
        print(f"✅ Statistics written in {self.statistics_output}")

    def write_stats_to_dataset(self, stats):
        if self.user_statistics_start or self.user_statistics_end:
            raise ValueError(
                (
                    "Cannot write statistics in dataset with user specified dates. "
                    "This would be conflicting with the dataset metadata."
                )
            )

        if not all(self.registry.get_flags(sync=False)):
            raise Exception(f"❗Zarr {self.path} is not fully built, not writting statistics into dataset.")

        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count"]:
            self._add_dataset(name=k, array=stats[k])

        self.registry.add_to_history("compute_statistics_end")
        print(f"Wrote statistics in {self.path}")

    def write_stats_to_stdout(self, stats):
        print(stats)


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


class CleanupLoader(Loader):
    def run(self):
        self.statistics_registry.delete()
        self.registry.clean()
