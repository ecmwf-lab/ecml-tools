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


class Group(list):
    """Interface wrapper for List objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self) >= 1, self
        assert all(isinstance(_, datetime.datetime) for _ in self), self

    def __repr__(self):
        try:
            content = ",".join([str(_.strftime("%Y-%m-%d:%H")) for _ in self])
            return f"Group({len(self)}, {content})"
        except Exception:
            return super().__repr__()


class BaseGroups:
    def __repr__(self):
        try:
            content = "+".join([str(len(list(g))) for g in self.groups])
            print(content)
            for g in self.groups:
                assert isinstance(g[0], datetime.datetime), g[0]
            print("val", self.values, self.n_groups)
            return f"{self.__class__.__name__}({content}={len(self.values)})({self.n_groups} groups)"
        except:  # noqa
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

    @property
    def frequency(self):
        datetimes = self.values
        freq = (datetimes[1] - datetimes[0]).total_seconds() / 3600
        assert round(freq) == freq, freq
        assert int(freq) == freq, freq
        frequency = int(freq)
        return frequency


class Groups(BaseGroups):
    def __init__(self, config):
        # Assert config input is ad dict but not a nested dict
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dict. {config}")
        for k, v in config.items():
            if isinstance(v, dict):
                raise ValueError(f"Values can't be a dictionary. {k,v}")

        self._config = config

    @property
    def groups(self):
        # Return a list where each sublist contain the subgroups
        # of values according to the grouper_key
        return [
            Group(g) for _, g in itertools.groupby(self.values, key=self.grouper_key)
        ]

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

    @cached_property
    def n_groups(self):
        return len(self.groups)


class ExpandGroups(Groups):
    def __init__(self, config):
        super().__init__(config)

        def _(x):
            if isinstance(x, str):
                return to_datetime(x)
            return x

        self.values = [_(x) for x in self._config.get("values")]


class SingleGroup(Groups):
    def __init__(self, group):
        self.values = group

    @property
    def groups(self):
        return [Group(self.values)]


class DateStartStopGroups(Groups):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def start(self):
        return self._get_date("start")

    @cached_property
    def end(self):
        return self._get_date("end")

    def _get_date(self, date_key):
        date = self._config[date_key]
        if isinstance(date, str):
            try:
                # Attempt to parse the date string with timestamp format
                check_timestamp = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
                if check_timestamp:
                    return to_datetime(date)
            except ValueError:
                raise ValueError(
                    f"{date_key}  must include timestamp not just date {date,type(date)}"
                )
        elif type(date) == datetime.date:  # noqa: E721
            raise ValueError(
                f"{date_key}  must include timestamp not just date {date,type(date)}"
            )
        else:
            return date

    def _validate_date_range(self):
        assert (
            self.end >= self.start
        ), "End date must be greater than or equal to start date."

    def _extract_frequency(self, frequency_str):
        freq_ending = frequency_str.lower()[-1]
        freq_mapping = {"h": int(frequency_str[:-1]), "d": int(frequency_str[:-1]) * 24}
        try:
            return freq_mapping[freq_ending]
        except:  # noqa: E722
            raise ValueError(
                f"Frequency must be in hours or days (12h or 2d). {frequency_str}"
            )

    def _validate_frequency(self, freq, frequency_str):
        if freq > 24 and freq % 24 != 0:
            raise ValueError(
                f"Frequency must be less than 24h or a multiple of 24h. {frequency_str}"
            )

    @cached_property
    def step(self):
        _frequency_str = self._config.get("frequency", "24h")
        _freq = self._extract_frequency(_frequency_str)
        self._validate_frequency(_freq, _frequency_str)
        return datetime.timedelta(hours=_freq)

    @cached_property
    def values(self):
        x = self.start
        dates = []
        while x <= self.end:
            dates.append(x)

            x += self.step
        assert isinstance(dates[0], datetime.datetime), dates[0]
        return dates


class EmptyGroups(BaseGroups):
    def __init__(self):
        self.values = []
        self.groups = []

    @property
    def frequency(self):
        return None


class GroupsIntersection(BaseGroups):
    def __init__(self, a, b):
        assert isinstance(a, Groups), a
        assert isinstance(b, Groups), b
        self.a = a
        self.b = b

    @cached_property
    def values(self):
        return list(set(self.a.values) & set(self.b.values))


def build_groups(config):
    if isinstance(config, Group):
        return SingleGroup(config)

    assert isinstance(config, dict), config

    if "values" in config:
        return ExpandGroups(config)

    if "start" in config and "end" in config:
        return DateStartStopGroups(config)

    raise NotImplementedError(config)
