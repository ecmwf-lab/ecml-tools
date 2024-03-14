# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .forewards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Select(Forwards):
    """Select a subset of the variables."""

    def __init__(self, dataset, indices, title):
        while isinstance(dataset, Select):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)
        assert len(self.indices) > 0
        self.title = title or {"indices": self.indices}

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

    def tree(self):
        return Node(self, [self.dataset.tree()], **self.title)


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

    def tree(self):
        return Node(self, [self.forward.tree()], rename=self.rename)
