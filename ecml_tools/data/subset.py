# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .forewards import Combined
from .forewards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import make_slice_or_index_from_list_or_tuple
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Join(Combined):
    """Join the datasets along the variables axis."""

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

        from .select import Select

        return Select(self, indices, {"overlay": True})

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
            k: np.concatenate([d.statistics[k] for d in self.datasets], axis=0) for k in self.datasets[0].statistics
        }

    def source(self, index):
        i = index
        for dataset in self.datasets:
            if i < dataset.shape[1]:
                return Source(self, index, dataset.source(i))
            i -= dataset.shape[1]
        assert False

    @cached_property
    def missing(self):
        result = set()
        for d in self.datasets:
            result = result | d.missing
        return result

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])


class Subset(Forwards):
    """Select a subset of the dates."""

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

        assert n >= 0, n
        n = self.indices[n]
        return self.dataset[n]

    @debug_indexing
    def _get_slice(self, s):
        # TODO: check if the indices can be simplified to a slice
        # the time checking maybe be longer than the time saved
        # using a slice
        indices = [self.indices[i] for i in range(*s.indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        if isinstance(indices, slice):
            return self.dataset[indices]
        return np.stack([self.dataset[i] for i in indices])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n):
        index, changes = index_to_slices(n, self.shape)
        # print('INDEX', index, changes)
        indices = [self.indices[i] for i in range(*index[0].indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
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
        return f"Subset({self.dataset},{self.dates[0]}...{self.dates[-1]}/{self.frequency})"

    @cached_property
    def missing(self):
        return {self.indices[i] for i in self.dataset.missing if i in self.indices}

    def tree(self):
        return Node(self, [self.dataset.tree()])
