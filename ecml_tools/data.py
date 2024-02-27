# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import calendar
import datetime
import logging
import os
import re
import warnings
from functools import cached_property, wraps
from pathlib import PurePath

import numpy as np
import yaml
import zarr

import ecml_tools

from .indexing import (
    apply_index_to_slices_changes,
    expand_list_indexing,
    index_to_slices,
    length_to_slices,
    update_tuple,
)

LOG = logging.getLogger(__name__)

__all__ = ["open_dataset", "open_zarr", "debug_zarr_loading"]

DEBUG_ZARR_LOADING = int(os.environ.get("DEBUG_ZARR_LOADING", "0"))
DEBUG_ZARR_INDEXING = int(os.environ.get("DEBUG_ZARR_INDEXING", "0"))

DEPTH = 0


def _debug_indexing(method):
    @wraps(method)
    def wrapper(self, index):
        global DEPTH
        # if isinstance(index, tuple):
        print("  " * DEPTH, "->", self, method.__name__, index)
        DEPTH += 1
        result = method(self, index)
        DEPTH -= 1
        # if isinstance(index, tuple):
        print("  " * DEPTH, "<-", self, method.__name__, result.shape)
        return result

    return wrapper


if DEBUG_ZARR_INDEXING:
    debug_indexing = _debug_indexing
else:
    debug_indexing = lambda x: x  # noqa


def debug_zarr_loading(on_off):
    global DEBUG_ZARR_LOADING
    DEBUG_ZARR_LOADING = on_off


def _make_slice_or_index_from_list_or_tuple(indices):
    """
    Convert a list or tuple of indices to a slice or an index, if possible.
    """

    if len(indices) < 2:
        return indices

    step = indices[1] - indices[0]

    if step > 0 and all(
        indices[i] - indices[i - 1] == step for i in range(1, len(indices))
    ):
        return slice(indices[0], indices[-1] + step, step)

    return indices


class Dataset:
    arguments = {}

    @cached_property
    def _len(self):
        return len(self)

    def _subset(self, **kwargs):
        if not kwargs:
            return self

        if "start" in kwargs or "end" in kwargs:
            start = kwargs.pop("start", None)
            end = kwargs.pop("end", None)

            return Subset(self, self._dates_to_indices(start, end))._subset(**kwargs)

        if "frequency" in kwargs:
            frequency = kwargs.pop("frequency")
            return Subset(self, self._frequency_to_indices(frequency))._subset(**kwargs)

        if "select" in kwargs:
            select = kwargs.pop("select")
            return Select(self, self._select_to_columns(select))._subset(**kwargs)

        if "drop" in kwargs:
            drop = kwargs.pop("drop")
            return Select(self, self._drop_to_columns(drop))._subset(**kwargs)

        if "reorder" in kwargs:
            reorder = kwargs.pop("reorder")
            return Select(self, self._reorder_to_columns(reorder))._subset(**kwargs)

        if "rename" in kwargs:
            rename = kwargs.pop("rename")
            return Rename(self, rename)._subset(**kwargs)

        if "statistics" in kwargs:
            statistics = kwargs.pop("statistics")
            return Statistics(self, statistics)._subset(**kwargs)

        raise NotImplementedError("Unsupported arguments: " + ", ".join(kwargs))

    def _frequency_to_indices(self, frequency):
        requested_frequency = _frequency_to_hours(frequency)
        dataset_frequency = _frequency_to_hours(self.frequency)
        assert requested_frequency % dataset_frequency == 0
        # Question: where do we start? first date, or first date that is a multiple of the frequency?
        step = requested_frequency // dataset_frequency

        return range(0, len(self), step)

    def _dates_to_indices(self, start, end):
        # TODO: optimize

        start = self.dates[0] if start is None else _as_first_date(start, self.dates)
        end = self.dates[-1] if end is None else _as_last_date(end, self.dates)

        return [i for i, date in enumerate(self.dates) if start <= date <= end]

    def _select_to_columns(self, vars):
        if isinstance(vars, set):
            # We keep the order of the variables as they are in the zarr file
            nvars = [v for v in self.name_to_index if v in vars]
            assert len(nvars) == len(vars)
            return self._select_to_columns(nvars)

        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        return [self.name_to_index[v] for v in vars]

    def _drop_to_columns(self, vars):
        if not isinstance(vars, (list, tuple, set)):
            vars = [vars]

        assert set(vars) <= set(self.name_to_index)

        return sorted([v for k, v in self.name_to_index.items() if k not in vars])

    def _reorder_to_columns(self, vars):
        if isinstance(vars, (list, tuple)):
            vars = {k: i for i, k in enumerate(vars)}

        indices = []

        for k, v in sorted(vars.items(), key=lambda x: x[1]):
            indices.append(self.name_to_index[k])

        # Make sure we don't forget any variables
        assert set(indices) == set(range(len(self.name_to_index)))

        return indices

    def dates_interval_to_indices(self, start, end):
        return self._dates_to_indices(start, end)

    def provenance(self):
        return {}

    def sub_shape(self, drop_axis):
        shape = self.shape
        shape = list(shape)
        shape.pop(drop_axis)
        return tuple(shape)

    def metadata(self):
        def tidy(v):
            if isinstance(v, (list, tuple)):
                return [tidy(i) for i in v]
            if isinstance(v, dict):
                return {k: tidy(v) for k, v in v.items()}
            if isinstance(v, str) and v.startswith("/"):
                return os.path.basename(v)
            return v

        return tidy(
            dict(
                version=ecml_tools.__version__,
                shape=self.shape,
                arguments=self.arguments,
                specific=self.metadata_specific(),
                frequency=self.frequency,
                variables=self.variables,
            )
        )

    def metadata_specific(self, **kwargs):
        action = self.__class__.__name__.lower()
        assert isinstance(self.frequency, int), (self.frequency, self, action)
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            frequency=self.frequency,
            **kwargs,
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n):
        raise NotImplementedError(
            f"Tuple not supported: {n} (class {self.__class__.__name__})"
        )


class Source:
    """
    Class used to follow the provenance of a data point.
    """

    def __init__(self, dataset, index, source=None, info=None):
        self.dataset = dataset
        self.index = index
        self.source = source
        self.info = info

    def __repr__(self):
        p = s = self.source
        while s is not None:
            p = s
            s = s.source

        return (
            f"{self.dataset}[{self.index}, {self.dataset.variables[self.index]}] ({p})"
        )

    def target(self):
        p = s = self.source
        while s is not None:
            p = s
            s = s.source
        return p

    def dump(self, depth=0):
        print(" " * depth, self)
        if self.source is not None:
            self.source.dump(depth + 1)


class ReadOnlyStore(zarr.storage.BaseStore):
    def __delitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class HTTPStore(ReadOnlyStore):
    """
    We write our own HTTPStore because the one used by zarr (fsspec)
    does not play well with fork() and multiprocessing.
    """

    def __init__(self, url):
        self.url = url

    def __getitem__(self, key):
        import requests

        r = requests.get(self.url + "/" + key)

        if r.status_code == 404:
            raise KeyError(key)

        r.raise_for_status()
        return r.content


class S3Store(ReadOnlyStore):
    """
    We write our own S3Store because the one used by zarr (fsspec)
    does not play well with fork() and multiprocessing.
    """

    def __init__(self, url):
        import boto3

        self.bucket, self.key = url[5:].split("/", 1)

        # TODO: get the profile name from the url
        self.s3 = boto3.Session(profile_name=None).client("s3")

    def __getitem__(self, key):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key + "/" + key)
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(key)

        return response["Body"].read()


class DebugStore(ReadOnlyStore):
    def __init__(self, store):
        assert not isinstance(store, DebugStore)
        self.store = store

    def __getitem__(self, key):
        # print()
        print("GET", key, self)
        # traceback.print_stack(file=sys.stdout)
        return self.store[key]

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        warnings.warn("DebugStore: iterating over the store")
        return iter(self.store)

    def __contains__(self, key):
        return key in self.store


def open_zarr(path):
    try:
        store = path

        if store.startswith("http://") or store.startswith("https://"):
            store = HTTPStore(store)

        elif store.startswith("s3://"):
            store = S3Store(store)

        if DEBUG_ZARR_LOADING:
            if isinstance(store, str):
                store = zarr.storage.DirectoryStore(store)
            store = DebugStore(store)

        return zarr.convenience.open(store, "r")
    except Exception:
        LOG.exception("Failed to open %r", path)
        raise


class Zarr(Dataset):
    def __init__(self, path):
        if isinstance(path, zarr.hierarchy.Group):
            self.path = str(id(path))
            self.z = path
        else:
            self.path = str(path)
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z.data

    def __len__(self):
        return self.data.shape[0]

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n):
        return self.data[n]

    def _unwind(self, index, rest, shape, axis, axes):
        if not isinstance(index, (int, slice, list, tuple)):
            try:
                # NumPy arrays, TensorFlow tensors, etc.
                index = tuple(index.tolist())
                assert not isinstance(index, bool), "Mask not supported"
            except AttributeError:
                pass

        if isinstance(index, (list, tuple)):
            axes.append(axis)  # Dimension of the concatenation
            for i in index:
                yield from self._unwind((slice(i, i + 1),), rest, shape, axis, axes)
            return

        if len(rest) == 0:
            yield (index,)
            return

        for n in self._unwind(rest[0], rest[1:], shape, axis + 1, axes):
            yield (index,) + n

    @cached_property
    def chunks(self):
        return self.z.data.chunks

    @cached_property
    def shape(self):
        return self.data.shape

    @cached_property
    def dtype(self):
        return self.z.data.dtype

    @cached_property
    def dates(self):
        return self.z.dates[:]  # Convert to numpy

    @property
    def latitudes(self):
        try:
            return self.z.latitudes[:]
        except AttributeError:
            LOG.warning("No 'latitudes' in %r, trying 'latitude'", self)
            return self.z.latitude[:]

    @property
    def longitudes(self):
        try:
            return self.z.longitudes[:]
        except AttributeError:
            LOG.warning("No 'longitudes' in %r, trying 'longitude'", self)
            return self.z.longitude[:]

    @property
    def statistics(self):
        return dict(
            mean=self.z.mean[:],
            stdev=self.z.stdev[:],
            maximum=self.z.maximum[:],
            minimum=self.z.minimum[:],
        )

    @property
    def resolution(self):
        return self.z.attrs["resolution"]

    @property
    def frequency(self):
        try:
            return self.z.attrs["frequency"]
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'dates'", self)
        dates = self.dates
        delta = dates[1].astype(object) - dates[0].astype(object)
        return int(delta.total_seconds() / 3600)

    @property
    def name_to_index(self):
        if "variables" in self.z.attrs:
            return {n: i for i, n in enumerate(self.z.attrs["variables"])}
        return self.z.attrs["name_to_index"]

    @property
    def variables(self):
        return [
            k
            for k, v in sorted(
                self.name_to_index.items(),
                key=lambda x: x[1],
            )
        ]

    def __repr__(self):
        return self.path

    def end_of_statistics_date(self):
        return self.dates[-1]

    def metadata_specific(self):
        return super().metadata_specific(
            attrs=dict(self.z.attrs),
            chunks=self.chunks,
            dtype=str(self.dtype),
        )

    def source(self, index):
        return Source(self, index, info=self.path)


class Forwards(Dataset):
    def __init__(self, forward):
        self.forward = forward

    def __len__(self):
        return len(self.forward)

    def __getitem__(self, n):
        return self.forward[n]

    @property
    def dates(self):
        return self.forward.dates

    @property
    def resolution(self):
        return self.forward.resolution

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def latitudes(self):
        return self.forward.latitudes

    @property
    def longitudes(self):
        return self.forward.longitudes

    @property
    def name_to_index(self):
        return self.forward.name_to_index

    @property
    def variables(self):
        return self.forward.variables

    @property
    def statistics(self):
        return self.forward.statistics

    @property
    def shape(self):
        return self.forward.shape

    @property
    def dtype(self):
        return self.forward.dtype

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(
            forward=self.forward.metadata_specific(),
            **kwargs,
        )


class Combined(Forwards):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

        # Forward most properties to the first dataset
        super().__init__(datasets[0])

    def check_same_resolution(self, d1, d2):
        if d1.resolution != d2.resolution:
            raise ValueError(
                f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})"
            )

    def check_same_frequency(self, d1, d2):
        if d1.frequency != d2.frequency:
            raise ValueError(
                f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})"
            )

    def check_same_grid(self, d1, d2):
        if (d1.latitudes != d2.latitudes).any() or (
            d1.longitudes != d2.longitudes
        ).any():
            raise ValueError(f"Incompatible grid ({d1} {d2})")

    def check_same_shape(self, d1, d2):
        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

        if d1.variables != d2.variables:
            raise ValueError(
                f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})"
            )

    def check_same_sub_shapes(self, d1, d2, drop_axis):
        shape1 = d1.sub_shape(drop_axis)
        shape2 = d2.sub_shape(drop_axis)

        if shape1 != shape2:
            raise ValueError(
                f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})"
            )

    def check_same_variables(self, d1, d2):
        if d1.variables != d2.variables:
            raise ValueError(
                f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})"
            )

    def check_same_lengths(self, d1, d2):
        if d1._len != d2._len:
            raise ValueError(f"Incompatible lengths: {d1._len} and {d2._len}")

    def check_same_dates(self, d1, d2):
        self.check_same_frequency(d1, d2)

        if d1.dates[0] != d2.dates[0]:
            raise ValueError(
                f"Incompatible start dates: {d1.dates[0]} and {d2.dates[0]} ({d1} {d2})"
            )

        if d1.dates[-1] != d2.dates[-1]:
            raise ValueError(
                f"Incompatible end dates: {d1.dates[-1]} and {d2.dates[-1]} ({d1} {d2})"
            )

    def check_compatibility(self, d1, d2):
        # These are the default checks
        # Derived classes should turn individual checks off if they are not needed
        self.check_same_resolution(d1, d2)
        self.check_same_frequency(d1, d2)
        self.check_same_grid(d1, d2)
        self.check_same_lengths(d1, d2)
        self.check_same_variables(d1, d2)
        self.check_same_dates(d1, d2)

    def provenance(self):
        return [d.provenance() for d in self.datasets]

    def __repr__(self):
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"{self.__class__.__name__}({lst})"

    def metadata_specific(self, **kwargs):
        # We need to skip the forward superclass
        # TODO: revisit this
        return Dataset.metadata_specific(
            self,
            datasets=[d.metadata_specific() for d in self.datasets],
            **kwargs,
        )


class Concat(Combined):
    def __len__(self):
        return sum(len(i) for i in self.datasets)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        # print(index, changes)
        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(index[0], lengths)
        # print("slies", slices)
        result = [
            d[update_tuple(index, 0, i)[0]]
            for (d, i) in zip(self.datasets, slices)
            if i is not None
        ]
        result = np.concatenate(result, axis=0)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        # TODO: optimize
        k = 0
        while n >= self.datasets[k]._len:
            n -= self.datasets[k]._len
            k += 1
        return self.datasets[k][n]

    @debug_indexing
    def _get_slice(self, s):
        result = []

        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(s, lengths)

        result = [d[i] for (d, i) in zip(self.datasets, slices) if i is not None]

        return np.concatenate(result)

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def check_same_lengths(self, d1, d2):
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1, d2):
        # Turned off because we are concatenating along the dates axis
        pass

    @property
    def dates(self):
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self):
        return (len(self),) + self.datasets[0].shape[1:]


class GivenAxis(Combined):
    """Given a given axis, combine the datasets along that axis"""

    def __init__(self, datasets, axis):
        self.axis = axis
        super().__init__(datasets)

        assert axis > 0 and axis < len(self.datasets[0].shape), (
            axis,
            self.datasets[0].shape,
        )

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=self.axis)

    @cached_property
    def shape(self):
        shapes = [d.shape for d in self.datasets]
        before = shapes[0][: self.axis]
        after = shapes[0][self.axis + 1 :]
        result = before + (sum(s[self.axis] for s in shapes),) + after
        assert False not in result, result
        return result

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        lengths = [d.shape[self.axis] for d in self.datasets]
        slices = length_to_slices(index[self.axis], lengths)
        result = [
            d[update_tuple(index, self.axis, i)[0]]
            for (d, i) in zip(self.datasets, slices)
            if i is not None
        ]
        result = np.concatenate(result, axis=self.axis)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return np.concatenate([d[n] for d in self.datasets], axis=self.axis - 1)


class Ensemble(GivenAxis):
    pass


class Grids(GivenAxis):
    def __init__(self, datasets, axis):
        super().__init__(datasets, axis)
        # Shape: (dates, variables, ensemble, 1d-values)
        assert len(datasets[0].shape) == 4, "Grids must be 1D for now"

    def check_same_grid(self, d1, d2):
        # We don't check the grid, because we want to be able to combine
        pass


class ConcatGrids(Grids):
    # TODO: select the statistics of the most global grid?
    @property
    def latitudes(self):
        return np.concatenate([d.latitudes for d in self.datasets])

    @property
    def longitudes(self):
        return np.concatenate([d.longitudes for d in self.datasets])


class CutoutGrids(Grids):
    def __init__(self, datasets, axis):
        from .grids import cutout_mask

        super().__init__(datasets, axis)
        assert len(datasets) == 2, "CutoutGrids requires two datasets"
        assert axis == 3, "CutoutGrids requires axis=3"

        # We assume that the LAM is the first dataset, and the global is the second
        # Note: the second fields does not really need to be global

        self.lam, self.globe = datasets
        self.mask = cutout_mask(
            self.lam.latitudes,
            self.lam.longitudes,
            self.globe.latitudes,
            self.globe.longitudes,
            plot="cutout",
        )
        assert len(self.mask) == self.globe.shape[3], (
            len(self.mask),
            self.globe.shape[3],
        )

    @cached_property
    def shape(self):
        shape = self.lam.shape
        # Number of non-zero masked values in the globe dataset
        nb_globe = np.count_nonzero(self.mask)
        return shape[:-1] + (shape[-1] + nb_globe,)

    def check_same_resolution(self, d1, d2):
        # Turned off because we are combining different resolutions
        pass

    @property
    def latitudes(self):
        return np.concatenate([self.lam.latitudes, self.globe.latitudes[self.mask]])

    @property
    def longitudes(self):
        return np.concatenate([self.lam.longitudes, self.globe.longitudes[self.mask]])

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        assert index[self.axis] == slice(
            None
        ), "No support for selecting a subset of the 1D values"
        index, changes = index_to_slices(index, self.shape)

        # In case index_to_slices has changed the last slice
        index, _ = update_tuple(index, self.axis, slice(None))

        lam_data = self.lam[index]
        globe_data = self.globe[index]

        globe_data = globe_data[:, :, :, self.mask]

        result = np.concatenate([lam_data, globe_data], axis=self.axis)

        return apply_index_to_slices_changes(result, changes)


class Join(Combined):
    """
    Join the datasets along the variables axis.
    """

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=1)

    def check_same_variables(self, d1, d2):
        # Turned off because we are joining along the variables axis
        pass

    def __len__(self):
        return len(self.datasets[0])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))

        # TODO: optimize if index does not access all datasets, so we don't load chunks we don't need
        result = [d[index] for d in self.datasets]

        result = np.concatenate(result, axis=1)
        return apply_index_to_slices_changes(result[:, previous], changes)

    @debug_indexing
    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return np.concatenate([d[n] for d in self.datasets])

    @cached_property
    def shape(self):
        cols = sum(d.shape[1] for d in self.datasets)
        return (len(self), cols) + self.datasets[0].shape[2:]

    def _overlay(self):
        indices = {}
        i = 0
        for d in self.datasets:
            for v in d.variables:
                indices[v] = i
                i += 1

        if len(indices) == i:
            # No overlay
            return self

        indices = list(indices.values())

        i = 0
        for d in self.datasets:
            ok = False
            for v in d.variables:
                if i in indices:
                    ok = True
                i += 1
            if not ok:
                LOG.warning("Dataset %r completely overridden.", d)

        return Select(self, indices)

    @cached_property
    def variables(self):
        seen = set()
        result = []
        for d in reversed(self.datasets):
            for v in reversed(d.variables):
                while v in seen:
                    v = f"({v})"
                seen.add(v)
                result.insert(0, v)

        return result

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    @property
    def statistics(self):
        return {
            k: np.concatenate([d.statistics[k] for d in self.datasets], axis=0)
            for k in self.datasets[0].statistics
        }

    def source(self, index):
        i = index
        for dataset in self.datasets:
            if i < dataset.shape[1]:
                return Source(self, index, dataset.source(i))
            i -= dataset.shape[1]
        assert False


class Subset(Forwards):
    """
    Select a subset of the dates.
    """

    def __init__(self, dataset, indices):
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)

        # Forward other properties to the super dataset
        super().__init__(dataset)

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        n = self.indices[n]
        return self.dataset[n]

    @debug_indexing
    def _get_slice(self, s):
        # TODO: check if the indices can be simplified to a slice
        # the time checking maybe be longer than the time saved
        # using a slice
        indices = [self.indices[i] for i in range(*s.indices(self._len))]
        return np.stack([self.dataset[i] for i in indices])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n):
        index, changes = index_to_slices(n, self.shape)
        # print('INDEX', index, changes)
        indices = [self.indices[i] for i in range(*index[0].indices(self._len))]
        indices = _make_slice_or_index_from_list_or_tuple(indices)
        # print('INDICES', indices)
        index, _ = update_tuple(index, 0, indices)
        result = self.dataset[index]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def __len__(self):
        return len(self.indices)

    @cached_property
    def shape(self):
        return (len(self),) + self.dataset.shape[1:]

    @cached_property
    def dates(self):
        return self.dataset.dates[self.indices]

    @cached_property
    def frequency(self):
        dates = self.dates
        delta = dates[1].astype(object) - dates[0].astype(object)
        return int(delta.total_seconds() / 3600)

    def source(self, index):
        return Source(self, index, self.forward.source(index))

    def __repr__(self):
        return f"Subset({self.dates[0]}, {self.dates[-1]}, {self.frequency})"


class Select(Forwards):
    """
    Select a subset of the variables.
    """

    def __init__(self, dataset, indices):
        while isinstance(dataset, Select):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)
        assert len(self.indices) > 0

        # Forward other properties to the main dataset
        super().__init__(dataset)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.dataset[index]
        result = result[:, self.indices]
        result = result[:, previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        row = self.dataset[n]
        if isinstance(n, slice):
            return row[:, self.indices]

        return row[self.indices]

    @cached_property
    def shape(self):
        return (len(self), len(self.indices)) + self.dataset.shape[2:]

    @cached_property
    def variables(self):
        return [self.dataset.variables[i] for i in self.indices]

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    @cached_property
    def statistics(self):
        return {k: v[self.indices] for k, v in self.dataset.statistics.items()}

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(indices=self.indices, **kwargs)

    def source(self, index):
        return Source(self, index, self.dataset.source(self.indices[index]))


class Rename(Forwards):
    def __init__(self, dataset, rename):
        super().__init__(dataset)
        for n in rename:
            assert n in dataset.variables
        self._variables = [rename.get(v, v) for v in dataset.variables]
        self.rename = rename

    @property
    def variables(self):
        return self._variables

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(rename=self.rename, **kwargs)


class Statistics(Forwards):
    def __init__(self, dataset, statistic):
        super().__init__(dataset)
        self._statistic = statistic

    @cached_property
    def statistics(self):
        return open_dataset(self._statistic).statistics

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(
            statistics=open_dataset(self._statistic).metadata_specific(),
            **kwargs,
        )


def _name_to_path(name, zarr_root):
    _, ext = os.path.splitext(name)

    if ext in (".zarr", ".zip"):
        return name

    if zarr_root is None:
        with open(os.path.expanduser("~/.ecml-tools")) as f:
            zarr_root = yaml.safe_load(f)["zarr_root"]

    return os.path.join(zarr_root, name + ".zarr")


def _frequency_to_hours(frequency):
    if isinstance(frequency, int):
        return frequency

    if isinstance(frequency, float):
        assert int(frequency) == frequency
        return int(frequency)

    m = re.match(r"(\d+)([dh])?", frequency)
    if m is None:
        raise ValueError("Invalid frequency: " + frequency)

    frequency = int(m.group(1))
    if m.group(2) == "h":
        return frequency

    if m.group(2) == "d":
        return frequency * 24

    raise NotImplementedError()


def _as_date(d, dates, last):
    if isinstance(d, datetime.datetime):
        if not d.minute == 0 and d.hour == 0 and d.second == 0:
            return np.datetime64(d)
        d = datetime.date(d.year, d.month, d.day)

    if isinstance(d, datetime.date):
        d = d.year * 10_000 + d.month * 100 + d.day

    try:
        d = int(d)
    except (ValueError, TypeError):
        pass

    if isinstance(d, int):
        if len(str(d)) == 4:
            year = d
            if last:
                return np.datetime64(f"{year:04}-12-31T23:59:59")
            else:
                return np.datetime64(f"{year:04}-01-01T00:00:00")

        if len(str(d)) == 6:
            year = d // 100
            month = d % 100
            if last:
                _, last_day = calendar.monthrange(year, month)
                return np.datetime64(f"{year:04}-{month:02}-{last_day:02}T23:59:59")
            else:
                return np.datetime64(f"{year:04}-{month:02}-01T00:00:00")

        if len(str(d)) == 8:
            year = d // 10000
            month = (d % 10000) // 100
            day = d % 100
            if last:
                return np.datetime64(f"{year:04}-{month:02}-{day:02}T23:59:59")
            else:
                return np.datetime64(f"{year:04}-{month:02}-{day:02}T00:00:00")

    if isinstance(d, str):
        if "-" in d:
            assert ":" not in d
            bits = d.split("-")
            if len(bits) == 1:
                return _as_date(int(bits[0]), dates, last)

            if len(bits) == 2:
                return _as_date(int(bits[0]) * 100 + int(bits[1]), dates, last)

            if len(bits) == 3:
                return _as_date(
                    int(bits[0]) * 10000 + int(bits[1]) * 100 + int(bits[2]),
                    dates,
                    last,
                )
        if ":" in d:
            assert len(d) == 5
            hour, minute = d.split(":")
            assert minute == "00"
            assert not last
            first = dates[0].astype(object)
            year = first.year
            month = first.month
            day = first.day

            return np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour}:00:00")

    raise NotImplementedError(f"Unsupported date: {d} ({type(d)})")


def _as_first_date(d, dates):
    return _as_date(d, dates, last=False)


def _as_last_date(d, dates):
    return _as_date(d, dates, last=True)


def _concat_or_join(datasets, kwargs):
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    # Study the dates
    ranges = [(d.dates[0].astype(object), d.dates[-1].astype(object)) for d in datasets]

    if len(set(ranges)) == 1:
        return Join(datasets)._overlay(), kwargs

    # Make sure the dates are disjoint
    for i in range(len(ranges)):
        r = ranges[i]
        for j in range(i + 1, len(ranges)):
            s = ranges[j]
            if r[0] <= s[0] <= r[1] or r[0] <= s[1] <= r[1]:
                raise ValueError(
                    f"Overlapping dates: {r} and {s} ({datasets[i]} {datasets[j]})"
                )

    # For now we should have the datasets in order with no gaps

    frequency = _frequency_to_hours(datasets[0].frequency)

    for i in range(len(ranges) - 1):
        r = ranges[i]
        s = ranges[i + 1]
        if r[1] + datetime.timedelta(hours=frequency) != s[0]:
            raise ValueError(
                "Datasets must be sorted by dates, with no gaps: "
                f"{r} and {s} ({datasets[i]} {datasets[i+1]})"
            )

    return Concat(datasets), kwargs


def _open(a, zarr_root):
    if isinstance(a, Dataset):
        return a

    if isinstance(a, zarr.hierarchy.Group):
        return Zarr(a)

    if isinstance(a, str):
        return Zarr(_name_to_path(a, zarr_root))

    if isinstance(a, PurePath):
        return Zarr(str(a))

    if isinstance(a, dict):
        return _open_dataset(zarr_root=zarr_root, **a)

    if isinstance(a, (list, tuple)):
        return _open_dataset(*a, zarr_root=zarr_root)

    raise NotImplementedError()


def _auto_adjust(datasets, kwargs):
    """Adjust the datasets for concatenation or joining based
    on parameters set to 'matching'"""

    if kwargs.get("adjust") == "matching":
        kwargs.pop("adjust")
        for p in ("select", "frequency", "start", "end"):
            kwargs[p] = "matching"

    adjust = {}

    if kwargs.get("select") == "matching":
        kwargs.pop("select")
        variables = None

        for d in datasets:
            if variables is None:
                variables = set(d.variables)
            else:
                variables &= set(d.variables)

        if len(variables) == 0:
            raise ValueError("No common variables")

        adjust["select"] = sorted(variables)

    if kwargs.get("start") == "matching":
        kwargs.pop("start")
        adjust["start"] = max(d.dates[0] for d in datasets).astype(object)

    if kwargs.get("end") == "matching":
        kwargs.pop("end")
        adjust["end"] = min(d.dates[-1] for d in datasets).astype(object)

    if kwargs.get("frequency") == "matching":
        kwargs.pop("frequency")
        adjust["frequency"] = max(d.frequency for d in datasets)

    if adjust:
        datasets = [d._subset(**adjust) for d in datasets]

    return datasets, kwargs


def _open_dataset(*args, zarr_root, **kwargs):
    sets = []
    for a in args:
        sets.append(_open(a, zarr_root))

    if "ensemble" in kwargs:
        if "grids" in kwargs:
            raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

        ensemble = kwargs.pop("ensemble")
        axis = kwargs.pop("axis", 2)
        assert len(args) == 0
        assert isinstance(ensemble, (list, tuple))

        datasets = [_open(e, zarr_root) for e in ensemble]
        datasets, kwargs = _auto_adjust(datasets, kwargs)

        return Ensemble(datasets, axis=axis)._subset(**kwargs)

    if "grids" in kwargs:
        if "ensemble" in kwargs:
            raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

        grids = kwargs.pop("grids")
        mode = kwargs.pop("mode", "concatenate")
        axis = kwargs.pop("axis", 3)
        assert len(args) == 0
        assert isinstance(grids, (list, tuple))

        KLASSES = {
            "concatenate": ConcatGrids,
            "cutout": CutoutGrids,
        }
        if mode not in KLASSES:
            raise ValueError(
                f"Unknown grids mode: {mode}, values are {list(KLASSES.keys())}"
            )

        datasets = [_open(e, zarr_root) for e in grids]
        datasets, kwargs = _auto_adjust(datasets, kwargs)

        return KLASSES[mode](datasets, axis=axis)._subset(**kwargs)

    for name in ("datasets", "dataset"):
        if name in kwargs:
            datasets = kwargs.pop(name)
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            for a in datasets:
                sets.append(_open(a, zarr_root))

    assert len(sets) > 0, (args, kwargs)

    if len(sets) > 1:
        dataset, kwargs = _concat_or_join(sets, kwargs)
        return dataset._subset(**kwargs)

    return sets[0]._subset(**kwargs)


def open_dataset(*args, zarr_root=None, **kwargs):
    ds = _open_dataset(*args, zarr_root=zarr_root, **kwargs)
    ds.arguments = {"args": args, "kwargs": kwargs}
    return ds
