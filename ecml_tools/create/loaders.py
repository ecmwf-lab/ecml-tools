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
from .config import OutputSpecs, build_output, loader_config
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
)
from .writer import CubesFilter, DataWriter
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


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

    @classmethod
    def from_config(cls, *, config, path, print=print, **kwargs):
        return cls(config=config, path=path, print=print, **kwargs)

    @cached_property
    def registry(self):
        from .zarr import ZarrBuiltRegistry

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

        z = zarr.open(self.path, mode=mode)
        return add_zarr_dataset(zarr_root=z, **kwargs)

    def get_zarr_chunks(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        return z["data"].chunks
        assert chunks == z.data.chunks


class InitialiseLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.statistics_registry.delete()

        self.groups = build_groups(*self.main_config.loop)
        self.output = build_output(self.main_config.output, parent=self)
        self.inputs = build_input(self.main_config.input, parent=self, selection=None)

        print(self.groups)
        print(self.inputs)
        # print([(k,len(v)) for k,v in self.inputs.coords.items()])

    def initialise(self, check_name=True):
        """Create empty dataset"""

        self.print("Config loaded ok:")
        print(self.main_config)
        print("-------------------------")

        dates = self.inputs.dates_obj.values
        frequency = self.inputs.frequency
        assert isinstance(frequency, int), frequency

        firstdate = self.inputs.dates_obj.values[0]
        minimal_input = self.inputs.select(dates=firstdate)
        print(minimal_input)

        self.print(f"Found {len(dates)} datetimes.")
        print(
            f"Dates: Found {len(dates)} datetimes, in {self.groups.n_groups} groups: ",
            end="",
        )
        lengths = [str(len(g)) for g in self.groups]
        print("+".join(lengths))
        self.print(f"Found {len(dates)} datetimes {'+'.join(lengths)}.")
        print("-------------------------")

        variables = minimal_input.variables
        self.print(f"Found {len(variables)} variables : {','.join(variables)}.")

        ensembles = minimal_input.ensembles
        self.print(
            f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}."
        )

        grid_points = minimal_input.grid_points
        print(f"gridpoints size: {[len(i) for i in grid_points]}")
        print("-------------------------")

        resolution = minimal_input.resolution
        print(f"{resolution=}")

        print("-------------------------")
        coords = minimal_input.coords
        coords["dates"] = dates
        total_shape = minimal_input.shape
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
        metadata["resolution"] = resolution

        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        print("‼️‼️‼️‼️ no data_request in output zarr yet")
        # metadata["data_request"] = self.inputs.get_data_request()

        metadata["order_by"] = self.output.order_by_as_list
        metadata["remapping"] = self.inputs.remapping
        metadata["flatten_grid"] = self.output.flatten_grid
        metadata["ensemble_dimension"] = len(ensembles)

        metadata["variables"] = variables
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
        self.registry.add_to_history(
            "statistics_registry_initialised", version=self.statistics_registry.version
        )

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())

        print("‼️‼️‼️‼️ no data_request in output zarr yet")


class ContentLoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.groups = build_groups(*self.main_config.loop)
        self.output = build_output(self.main_config.output, parent=self)
        self.inputs = build_input(self.main_config.input, parent=self, selection=None)
        self.read_dataset_metadata()

    def load(self, parts):
        self.registry.add_to_history("loading_data_start", parts=parts)

        z = zarr.open(self.path, mode="r+")
        data_writer = DataWriter(
            parts, parent=self, full_array=z["data"], print=self.print
        )

        total = len(self.registry.get_flags())
        filter = CubesFilter(parts=parts, total=total)
        for igroup, group in enumerate(self.groups):
            print("✅", igroup, group)
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(group)} (already done)")
                continue
            if not filter(igroup):
                continue
            self.print(f" -> Processing i={igroup} total={len(group)}")
            assert isinstance(group[0], datetime.datetime), group

            inputs = self.inputs.select(dates=group)
            data_writer.write(inputs, igroup)
        exit()

        self.registry.add_to_history("loading_data_end", parts=parts)
        self.registry.add_provenance(name="provenance_load")
        self.statistics_registry.add_provenance(
            name="provenance_load", config=self.main_config
        )


class GenericStatisticsLoader(Loader):
    def __init__(
        self,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @property
    def _incomplete(self):
        return not all(self.registry.get_flags(sync=False))

    def read_dataset_dates_metadata(self, statistics_start, statistics_end):
        ds = open_dataset(self.path)

        subset = ds.dates_interval_to_indices(statistics_start, statistics_end)
        self.i_start = subset[0]
        self.i_end = subset[-1]
        self.date_start = ds.dates[subset[0]]
        self.date_end = ds.dates[subset[-1]]

        def check():
            i_len = self.i_end + 1 - self.i_start
            self.print(f"Statistics computed on {i_len}/{len(ds.dates)} samples ")
            print(f"Requested ({i_len}): from {self.date_start} to {self.date_end}.")
            print(f"Available ({len(ds.dates)}): from {ds.dates[0]} to {ds.dates[-1]}.")
            if i_len < 1:
                raise ValueError("Cannot compute statistics on an empty interval.")

        check()

    def run(self):
        detailed = self.get_detailed_stats()
        self.write_detailed_statistics(detailed)

        selected = {k: v[self.i_start : self.i_end + 1] for k, v in detailed.items()}
        stats = compute_aggregated_statistics(selected, self.variables_names)
        self.write_aggregated_statistics(stats)

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
        if self.statistics_output:
            print(
                "Not writting detailed statistics into dataset because option 'output' is set."
            )
            return
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


class RecomputeStatisticsLoader(GenericStatisticsLoader):
    def __init__(
        self,
        statistics_start=None,
        statistics_end=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        print(f"Recomputing temporary statistics from {self.statistics_registry}")
        print("not writting statistics into dataset.")
        self._write_to_dataset = False

        if self._incomplete:
            msg = f"❗Zarr {self.path} is not fully built"
            if force:
                print(msg)
            else:
                raise Exception(msg)

        start = statistics_start or self.main_config.output.get("statistics_start")
        end = statistics_end or self.main_config.output.get("statistics_end")
        self.read_dataset_metadata()
        self.read_dataset_dates_metadata(start, end)

    def get_detailed_stats(self):
        self._build_temporary_statistics()
        return super().get_detailed_stats()

    def _build_temporary_statistics(self):
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


class StatisticsLoader(GenericStatisticsLoader):
    def __init__(
        self,
        statistics_output=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._write_to_dataset = True

        self.statistics_output = statistics_output

        if self.statistics_output:
            self._write_to_dataset = False

        if self._incomplete:
            msg = f"❗Zarr {self.path} is not fully built"
            if force:
                print(msg)
                print("Not writting statistics into dataset.")
                self._write_to_dataset = False
            else:
                raise Exception(msg)

        start = self.main_config.output.get("statistics_start")
        end = self.main_config.output.get("statistics_end")
        self.read_dataset_metadata()
        self.read_dataset_dates_metadata(start, end)


class SizeLoader(Loader):
    def __init__(self, path, print):
        self.path = path
        self.print = print

    @classmethod
    def from_dataset(cls, path, print):
        return cls(path, print)

    def add_total_size(self):
        dic = compute_directory_sizes(self.path)

        size = dic["total_size"]
        n = dic["total_number_of_files"]

        print(f"Total size: {bytes(size)}")
        print(f"Total number of files: {n}")

        self.update_metadata(total_size=size, total_number_of_files=n)
