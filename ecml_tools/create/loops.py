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

from .utils import to_datetime


class Expand(list):
    """
    This class is used to expand loops.
    It creates a list of list in self.groups.
    """

    def __init__(self, config, **kwargs):
        self._config = config
        self.kwargs = kwargs
        self.groups = []
        self.parse_config()

    def parse_config(self):
        self.start = self._config.get("start")
        if self.start is not None:
            self.start = to_datetime(self.start)
        self.end = self._config.get("end", self._config.get("stop"))
        if self.end is not None:
            self.end = to_datetime(self.end)
        self.step = self._config.get("step", self._config.get("frequency", 1))
        self.group_by = self._config.get("group_by")


class ValuesExpand(Expand):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        values = self._config["values"]
        values = [[v] if not isinstance(v, list) else v for v in values]
        for v in self._config["values"]:
            if not isinstance(v, (tuple, list)):
                v = [v]
            self.groups.append(v)


class StartStopExpand(Expand):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        x = self.start
        all = []
        while x <= self.end:
            all.append(x)
            x += self.step

        result = [list(g) for _, g in itertools.groupby(all, key=self.grouper_key)]
        self.groups = [[self.format(x) for x in g] for g in result]

    def parse_config(self):
        if "stop" in self._config:
            raise ValueError(f"Use 'end' not 'stop' in loop. {self._config}")
        super().parse_config()

    def format(self, x):
        return x


class GroupByDays:
    def __init__(self, days):
        self.days = days

    def __call__(self, dt):
        year = dt.year
        days = (dt - datetime.datetime(year, 1, 1)).days
        x = (year, days // self.days)
        # print(x)
        return x


class DateStartStopExpand(StartStopExpand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_config(self):
        super().parse_config()
        assert isinstance(self.start, datetime.date), (type(self.start), self.start)
        assert isinstance(self.end, datetime.date), (type(self.end), self.end)
        self.step = datetime.timedelta(days=self.step)

        if isinstance(self.group_by, int) and self.group_by > 0:
            self.grouper_key = GroupByDays(self.group_by)
        else:
            self.grouper_key = {
                0: lambda dt: 0,  # only one group
                "monthly": lambda dt: (dt.year, dt.month),
                "daily": lambda dt: (dt.year, dt.month, dt.day),
                "MMDD": lambda dt: (dt.month, dt.day),
            }[self.group_by]

    def format(self, x):
        return x.isoformat()


class IntegerStartStopExpand(StartStopExpand):
    def grouper_key(self, x):
        return {
            1: lambda x: 0,  # only one group
            None: lambda x: x,  # one group per value
        }[self.group_by](x)


def _expand_class(values):
    if isinstance(values, list):
        values = {"values": values}

    assert isinstance(values, dict), values

    if isinstance(values.get("values"), list):
        assert len(values) == 1, f"No other config keys implemented. {values}"
        return ValuesExpand

    start = values.get("start")
    if start:
        if isinstance(start, datetime.datetime):
            return DateStartStopExpand
        if values.get("group_by") in [
            "monthly",
            "daily",
            "weekly",
            "fortnightly",
        ] or isinstance(values.get("group_by"), int):
            return DateStartStopExpand
        return IntegerStartStopExpand

    raise ValueError(f"Cannot expand loop from {values}")


def expand_loops(values, **kwargs):
    cls = _expand_class(values)
    return cls(values, **kwargs).groups
