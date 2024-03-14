# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import textwrap
from functools import wraps

LOG = logging.getLogger(__name__)

DEBUG_ZARR_LOADING = int(os.environ.get("DEBUG_ZARR_LOADING", "0"))
DEBUG_ZARR_INDEXING = int(os.environ.get("DEBUG_ZARR_INDEXING", "0"))

DEPTH = 0

# TODO: make numpy arrays read-only
# a.flags.writeable = False


class Node:
    def __init__(self, dataset, kids, **kwargs):
        self.dataset = dataset
        self.kids = kids
        self.kwargs = kwargs

    def _put(self, indent, result):

        def _spaces(indent):
            return " " * indent if indent else ""

        result.append(f"{_spaces(indent)}{self.dataset.__class__.__name__}")
        for k, v in self.kwargs.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(i) for i in v)
                v = textwrap.shorten(v, width=40, placeholder="...")
            result.append(f"{_spaces(indent+2)}{k}: {v}")
        for kid in self.kids:
            kid._put(indent + 2, result)

    def __repr__(self):
        result = []
        self._put(0, result)
        return "\n".join(result)


class Source:
    """Class used to follow the provenance of a data point."""

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

        return f"{self.dataset}[{self.index}, {self.dataset.variables[self.index]}] ({p})"

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
