# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import itertools
from functools import cached_property

from .expand import expand_class


def expand_values(config):
    cls = expand_class(config)
    values = cls(config).values
    # assert isinstance(values, list), values
    # assert not isinstance(values[0], list), values
    return values


def expand_groups(config):
    cls = _expand_class(config)
    groups = cls(config).groups
    assert isinstance(groups, list), groups
    assert isinstance(groups[0], list), groups
    return groups


class Group(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self) >= 1, self

    def __repr__(self):
        content = ",".join([str(_.strftime("%Y-%m-%d:%H")) for _ in self])
        # content = ",".join([str(_.strftime("%Y-%m-%d:%H")) for _ in self[:10]]) + "..."
        return f"Group({len(self)}, {content})"


class Groups:
    def __init__(self, config):
        assert isinstance(config, dict), config
        for k, v in config.items():
            assert not isinstance(v, dict), (k, v)

        self._config = config
        self.cls = expand_class(config)

    @property
    def values(self):
        dates = list(self.cls(self._config).values)
        assert isinstance(dates[0], datetime.datetime), dates[0]
        return dates

    def __iter__(self):
        for i in self.cls(self._config).groups:
            g = Group(i)
            yield g

    @cached_property
    def n_groups(self):
        return len([i for i in self])

    @property
    def frequency(self):
        datetimes = self.values
        freq = (datetimes[1] - datetimes[0]).total_seconds() / 3600
        assert round(freq) == freq, freq
        assert int(freq) == freq, freq
        frequency = int(freq)
        return frequency

    def __repr__(self):
        content = "+".join([str(len(list(g))) for g in self])
        for g in self:
            assert isinstance(g[0], datetime.datetime), g[0]

        return f"{self.__class__.__name__}({content}={len(self.values)})({self.n_groups} groups)"


def build_groups(*objs):
    if len(objs) > 1:
        raise NotImplementedError()
    obj = objs[0]

    if isinstance(obj, Groups):
        return obj

    if isinstance(obj, list):
        return Groups(dict(values=obj))

    assert isinstance(obj, dict), obj
    if "dates" in obj:
        assert len(obj) == 1, obj
        return Groups(dict(obj["dates"]))

    return Groups(obj)
