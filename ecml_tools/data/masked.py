# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

from ..grids import cropping_mask
from . import Forwards
from .dataset import Dataset
from .debug import Node
from .debug import debug_indexing
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Masked(Forwards):
    def __init__(self, forward, mask):
        super().__init__(forward)
        assert len(forward.shape) == 4, "Grids must be 1D for now"
        self.mask = mask
        self.axis = 3

    @cached_property
    def shape(self):
        return self.forward.shape[:-1] + (np.count_nonzero(self.mask),)

    @cached_property
    def latitudes(self):
        return self.forward.latitudes[self.mask]

    @cached_property
    def longitudes(self):
        return self.forward.longitudes[self.mask]

    @debug_indexing
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._get_tuple(index)

        result = self.forward[index]
        # We don't support subsetting the grid values
        assert result.shape[-1] == len(self.mask), (result.shape, len(self.mask))

        return result[..., self.mask]

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, self.axis, slice(None))
        result = self.forward[index]
        result = result[..., self.mask]
        result = result[..., previous]
        result = apply_index_to_slices_changes(result, changes)
        return result


class Thinning(Masked):
    def __init__(self, forward, thinning, method):
        self.thinning = thinning
        self.method = method

        latitudes = forward.latitudes.reshape(forward.field_shape)
        longitudes = forward.longitudes.reshape(forward.field_shape)
        latitudes = latitudes[::thinning, ::thinning].flatten()
        longitudes = longitudes[::thinning, ::thinning].flatten()

        mask = [lat in latitudes and lon in longitudes for lat, lon in zip(forward.latitudes, forward.longitudes)]
        mask = np.array(mask, dtype=bool)

        super().__init__(forward, mask)

    def tree(self):
        return Node(self, [self.forward.tree()], thinning=self.thinning, method=self.method)


class Cropping(Masked):
    def __init__(self, forward, bounding_box):
        self.bounding_box = bounding_box

        if isinstance(bounding_box, Dataset):
            north = np.amax(bounding_box.latitudes)
            south = np.amin(bounding_box.latitudes)
            east = np.amax(bounding_box.longitudes)
            west = np.amin(bounding_box.longitudes)
            bounding_box = (north, west, south, east)

        mask = cropping_mask(forward.latitudes, forward.longitudes, *bounding_box)

        super().__init__(forward, mask)

    def tree(self):
        return Node(self, [self.forward.tree()], bounding_box=self.bounding_box)
