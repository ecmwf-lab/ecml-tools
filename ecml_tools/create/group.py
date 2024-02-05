# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
from functools import cached_property
from .config import DictObj

from .expand import expand_class


class Group(list):
    """
    Interface wrapper for List objects
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self) >= 1, self

    def __repr__(self):
        content = ",".join([str(_.strftime("%Y-%m-%d:%H")) for _ in self])
        return f"Group({len(self)}, {content})"


class BaseGroups:
    def __repr__(self):
        try:
            content = "+".join([str(len(list(g))) for g in self.groups])
            for g in self.groups:
                assert isinstance(g[0], datetime.datetime), g[0]
            return f"{self.__class__.__name__}({content}={len(self.values)})({self.n_groups} groups)"
        except:
            return f"{self.__class__.__name__}({len(self.values)} dates)"

    @cached_property
    def values(self):
        raise NotImplementedError()

    def intersect(self, dates):
        if dates is None:
            return self
        # before creating GroupsIntersection
        # we make sure that dates it's also a Groups Instance
        if not isinstance(dates, Groups):
            dates = build_groups(dates)
        return GroupsIntersection(self, dates)

    def empty(self):
        return len(self.values) == 0

    @cached_property
    def n_groups(self):
        return len(self.groups)

    @property
    def frequency(self):
        print(self)
        datetimes = self.values
        freq = (datetimes[1] - datetimes[0]).total_seconds() / 3600
        assert round(freq) == freq, freq
        assert int(freq) == freq, freq
        frequency = int(freq)
        return frequency


class Groups(BaseGroups):
    def __init__(self, config):
        # Assert config input is ad dict but not a nested dict
        assert isinstance(config, dict), config
        for k, v in config.items():
            assert not isinstance(v, dict), (k, v)

        self._config = config
        self.cls = expand_class(config) #This is just an Object class (not instantiated)

    @cached_property
    def values(self) -> list:
        dates = list(self.cls(self._config).values)
        assert isinstance(dates[0], datetime.datetime), dates[0]
        return dates

    @property
    def groups(self):
        return [Group(g) for g in self.cls(self._config).groups]


class EmptyGroups(BaseGroups):
    def __init__(self):
        self.values = []
        self.groups = []

    @property
    def frequency(self):
        return None


class GroupsIntersection(BaseGroups):
    def __init__(self, a, b):
        assert isinstance(a,Groups), a #!TODO double-check if that assumption is right
        assert isinstance(b,Groups), b
        self.a = a
        self.b = b

    @cached_property
    def values(self):
        return list(set(self.a.values) & set(self.b.values))


def build_groups(*objs):
    if len(objs) > 1:
        raise NotImplementedError()
    obj = objs[0]

    type_actions = {
        GroupsIntersection: lambda x: x,
        Groups: lambda x: x,
        list: lambda x: Groups(dict(values=x)),
        Group: lambda x: Groups(dict(values=x)),
        DictObj: lambda x: Groups(dict(x["dates"])) if "dates" in x and len(x) == 1 else Groups(x),
        dict: lambda x: Groups(dict(x["dates"])) if "dates" in x and len(x) == 1 else Groups(x),
    }
    return type_actions[type(obj)](obj)