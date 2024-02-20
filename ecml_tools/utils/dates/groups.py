# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import itertools
from collections import defaultdict

from ecml_tools.utils.dates import Dates


class Groups:
    """
    >>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[0]
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 12, 0)]

    >>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[1]
    [datetime.datetime(2023, 1, 2, 0, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> g = Groups(group_by=3, start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=24)
    >>> len(list(g))
    2
    >>> len(list(g)[0])
    3
    >>> len(list(g)[1])
    2
    """

    def __init__(self, **kwargs):
        group_by = kwargs.pop("group_by")
        self.dates = Dates.from_config(**kwargs)
        self.grouper = Grouper.from_config(group_by)

    def __iter__(self):
        return self.grouper(self.dates)

    def __len__(self):
        return len(list(self.grouper(self.dates)))


class Grouper:
    @classmethod
    def from_config(cls, group_by):
        if isinstance(group_by, int) and group_by > 0:
            return GrouperByFixedSize(group_by)
        if group_by is None:
            return GrouperOneGroup()
        key = {
            "monthly": lambda dt: (dt.year, dt.month),
            "daily": lambda dt: (dt.year, dt.month, dt.day),
            "weekly": lambda dt: (dt.weekday(),),
            "MMDD": lambda dt: (dt.month, dt.day),
        }[group_by]
        return GrouperByKey(key)


class GrouperOneGroup(Grouper):
    def __call__(self, dates):
        yield dates.values


class GrouperByKey(Grouper):
    def __init__(self, key):
        self.key = key

    def __call__(self, dates):
        for _, g in itertools.groupby(dates, key=self.key):
            yield list(g)


class GrouperByFixedSize(Grouper):
    def __init__(self, size):
        self.size = size

    def __call__(self, dates):
        batch = []
        for d in dates:
            batch.append(d)
            if len(batch) == self.size:
                yield batch
                batch = []
        if batch:
            yield batch
