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

from .utils import to_datetime


class GroupByDays:
    def __init__(self, days):
        self.days = days

    def __call__(self, dt):
        year = dt.year
        days = (dt - datetime.datetime(year, 1, 1)).days
        x = (year, days // self.days)
        return x


class Expand(list):
    """
    This class is used to expand loops.
    It creates a list of list in self.groups.
    Flatten values are in self.values.
    """

    def __init__(self, config):
        assert isinstance(config, dict), config
        for k, v in config.items():
            assert not isinstance(v, dict), (k, v)
        self._config = config
        if "stop" in self._config:
            raise ValueError(f"Use 'end' not 'stop' in loop. {self._config}")

    @property
    def values(self):
        raise NotImplementedError()

    @property
    def groups(self):
        raise NotImplementedError(type(self))


class ValuesExpand(Expand):
    def __init__(self, config):
        super().__init__(config)
        if not isinstance(self._config, dict):
            raise ValueError(f"Config must be a dict. {self._config}")
        if not isinstance(self._config["values"], list):
            raise ValueError(f"Values must be a list. {self._config}")

    @property
    def values(self):
        return self._config["values"]


class StartStopExpand(Expand):
    def __init__(self, config):
        super().__init__(config)
        self.start = self._config["start"]
        self.end = self._config["end"]

    @property
    def values(self):
        x = self.start
        all = []
        while x <= self.end:
            all.append(x)
            yield x
            x += self.step

    def format(self, x):
        return x

    @cached_property
    def groups(self):
        groups = []
        for _, g in itertools.groupby(self.values, key=self.grouper_key):
            g = [self.format(x) for x in g]
            groups.append(g)
        return groups


class DateStartStopExpand(StartStopExpand):
    def __init__(self, config):
        super().__init__(config)

        self.start = to_datetime(self.start)
        self.end = to_datetime(self.end)
        assert isinstance(self.start, datetime.date), (type(self.start), self.start)
        assert isinstance(self.end, datetime.date), (type(self.end), self.end)

        frequency = self._config.get("frequency", "24h")

        if frequency.lower().endswith("h"):
            freq = int(frequency[:-1])
        elif frequency.lower().endswith("d"):
            freq = int(frequency[:-1]) * 24
        else:
            raise ValueError(
                f"Frequency must be in hours or days (12h or 2d). {frequency}"
            )

        if freq > 24 and freq % 24 != 0:
            raise ValueError(
                f"Frequency must be less than 24h or a multiple of 24h. {frequency}"
            )

        self.step = datetime.timedelta(hours=freq)

    @property
    def grouper_key(self):
        group_by = self._config.get("group_by")
        if isinstance(group_by, int) and group_by > 0:
            return GroupByDays(group_by)
        return {
            None: lambda dt: 0,  # only one group
            0: lambda dt: 0,  # only one group
            "monthly": lambda dt: (dt.year, dt.month),
            "daily": lambda dt: (dt.year, dt.month, dt.day),
            "weekly": lambda dt: (dt.weekday(),),
            "MMDD": lambda dt: (dt.month, dt.day),
        }[group_by]

    def format(self, x):
        assert isinstance(x, datetime.date), (type(x), x)
        return x


class IntegerStartStopExpand(StartStopExpand):
    def __init__(self, config):
        super().__init__(config)
        self.step = self._config.get("step", 1)
        assert isinstance(self.step, int), config
        assert isinstance(self.start, int), config
        assert isinstance(self.end, int), config

    def grouper_key(self, x):
        group_by = self._config["group_by"]
        return {
            1: lambda x: 0,  # only one group
            None: lambda x: x,  # one group per value
        }[group_by](x)


def expand_class(config):
    if isinstance(config, list):
        config = {"values": config}

    assert isinstance(config, dict), config

    if "start" not in config and not "values" in config:
        raise ValueError(f"Cannot expand loop from {config}")

    if isinstance(config.get("values"), list):
        assert len(config) == 1, f"No other config keys implemented. {config}"
        return ValuesExpand

    if (
        config.get("group_by")
        in [
            "monthly",
            "daily",
            "weekly",
        ]
        or isinstance(config["start"], datetime.datetime)
        or isinstance(config["end"], datetime.datetime)
        or "frequency" in config
        or (config.get("kind") == "dates" and "start" in config)
    ):
        return DateStartStopExpand

    if isinstance(config["start"], int) or isinstance(config["end"], int):
        return IntegerStartStopExpand

    raise ValueError(f"Cannot expand loop from {config}")
