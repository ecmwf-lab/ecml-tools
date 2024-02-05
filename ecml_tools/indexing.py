# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np


def _tuple_with_slices(t, shape):
    """
    Replace all integers in a tuple with slices, so we preserve the dimensionality.
    """

    result = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in t)
    changes = tuple(j for (j, i) in enumerate(t) if isinstance(i, int))
    result = tuple(slice(*s.indices(shape[i])) for (i, s) in enumerate(result))

    return result, changes


def _extend_shape(index, shape):
    if Ellipsis in index:
        if index.count(Ellipsis) > 1:
            raise IndexError("Only one Ellipsis is allowed")
        ellipsis_index = index.index(Ellipsis)
        index = list(index)
        index[ellipsis_index] = slice(None)
        while len(index) < len(shape):
            index.insert(ellipsis_index, slice(None))
        index = tuple(index)

    while len(index) < len(shape):
        index = index + (slice(None),)

    return index


def _index_to_tuple(index, shape):
    if isinstance(index, int):
        return _extend_shape((index,), shape)
    if isinstance(index, slice):
        return _extend_shape((index,), shape)
    if isinstance(index, tuple):
        return _extend_shape(index, shape)
    if index is Ellipsis:
        return _extend_shape((Ellipsis,), shape)
    raise ValueError(f"Invalid index: {index}")


def index_to_slices(index, shape):
    """
    Convert an index to a tuple of slices, with the same dimensionality as the shape.
    """
    return _tuple_with_slices(_index_to_tuple(index, shape), shape)


def apply_index_to_slices_changes(result, changes):
    if changes:
        shape = result.shape
        for i in changes:
            assert shape[i] == 1, (i, changes, shape)
        result = np.squeeze(result, axis=changes)
    return result


def update_tuple(t, index, value):
    """
    Replace the elements of a tuple at the given index with a new value.
    """
    t = list(t)
    prev = t[index]
    t[index] = value
    return tuple(t), prev


def length_to_slices(index, lengths):
    """
    Convert an index to a list of slices, given the lengths of the dimensions.
    """
    total = sum(lengths)
    start, stop, step = index.indices(total)
    print(start, stop, step)

    # TODO: combine loops
    p = []
    pos = 0
    for length in lengths:
        end = pos + length
        p.append((pos, end))
        pos = end

    result = []

    for i, (s, e) in enumerate(p):
        pos = s
        if s % step:
            s = s + step - s % step
            assert s % step == 0
            assert s >= pos
        if max(s, start) <= min(e, stop):
            result.append((i, slice(s - pos, e - pos, step)))
        else:
            result.append(None)

    return result


class IndexTester:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        return index_to_slices(index, self.shape)


if __name__ == "__main__":
    t = IndexTester((1000, 8, 10, 20000))
    i = t[0, 1, 2, 3]
    print(i)

    # print(t[0])
    # print(t[0, 1, 2, 3])
    # print(t[0:10])
    # print(t[...])
    # print(t[:-1])
