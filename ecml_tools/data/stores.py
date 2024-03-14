# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from functools import cached_property

import numpy as np
import zarr

from . import MissingDate
from .dataset import Dataset
from .debug import DEBUG_ZARR_LOADING
from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .indexing import expand_list_indexing

LOG = logging.getLogger(__name__)


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
    """We write our own HTTPStore because the one used by zarr (fsspec) does not play
    well with fork() and multiprocessing."""

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
    """We write our own S3Store because the one used by zarr (fsspec) does not play well
    with fork() and multiprocessing."""

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
            self.was_zarr = True
            self.path = str(id(path))
            self.z = path
        else:
            self.was_zarr = False
            self.path = str(path)
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z.data
        self.missing = set()

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
    def field_shape(self):
        return tuple(self.z.attrs["field_shape"])

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

    def mutate(self):
        if len(self.z.attrs.get("missing_dates", [])):
            LOG.warn(f"Dataset {self} has missing dates")
            return ZarrWithMissingDates(self.z if self.was_zarr else self.path)
        return self

    def tree(self):
        return Node(self, [], path=self.path)


class ZarrWithMissingDates(Zarr):
    def __init__(self, path):
        super().__init__(path)

        missing_dates = self.z.attrs.get("missing_dates", [])
        missing_dates = [np.datetime64(x) for x in missing_dates]
        self.missing_to_dates = {i: d for i, d in enumerate(self.dates) if d in missing_dates}
        self.missing = set(self.missing_to_dates)

    def mutate(self):
        return self

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n):
        if isinstance(n, int):
            if n in self.missing:
                self._report_missing(n)
            return self.data[n]

        if isinstance(n, slice):
            common = set(range(*n.indices(len(self)))) & self.missing
            if common:
                self._report_missing(list(common)[0])
            return self.data[n]

        if isinstance(n, tuple):
            first = n[0]
            if isinstance(first, int):
                if first in self.missing:
                    self._report_missing(first)
                return self.data[n]

            if isinstance(first, slice):
                common = set(range(*first.indices(len(self)))) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.data[n]

            if isinstance(first, (list, tuple)):
                common = set(first) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.data[n]

        raise TypeError(f"Unsupported index {n} {type(n)}")

    def _report_missing(self, n):
        raise MissingDate(f"Date {self.missing_to_dates[n]} is missing (index={n})")

    def tree(self):
        return Node(self, [], path=self.path, missing=sorted(self.missing))
