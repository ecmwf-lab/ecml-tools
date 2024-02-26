# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime


def frequency_to_hours(frequency):
    if isinstance(frequency, int):
        return frequency
    assert isinstance(frequency, str), (type(frequency), frequency)

    unit = frequency[-1].lower()
    v = int(frequency[:-1])
    return {"h": v, "d": v * 24}[unit]


class Dates:
    """Base class for date generation.

    >>> Dates.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-02 00:00", "frequency": "1d"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)]

    >>> Dates.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-03 00:00", "frequency": "18h"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 18, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> Dates.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=6).as_dict()
    {'start': '2023-01-01T00:00:00', 'end': '2023-01-02T00:00:00', 'frequency': '6h'}

    >>> len(Dates.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=12))
    3
    """

    @classmethod
    def from_config(cls, **kwargs):
        if "values" in kwargs:
            return ValuesDates(**kwargs)
        return StartEndDates(**kwargs)

    def __iter__(self):
        for v in self.values:
            yield v

    def __getitem__(self, i):
        return self.values[i]

    def __len__(self):
        return len(self.values)

    @property
    def summary(self):
        return f"📅 {self.values[0]} ... {self.values[-1]}"


class ValuesDates(Dates):
    def __init__(self, values, **kwargs):
        self.values = sorted(values)
        assert not kwargs, f"Unexpected arguments {kwargs}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"values": self.values[0]}


class StartEndDates(Dates):
    def __init__(self, start, end, frequency=1, **kwargs):
        assert not kwargs, f"Unexpected arguments {kwargs}"

        frequency = frequency_to_hours(frequency)

        def _(x):
            if isinstance(x, str):
                return datetime.datetime.fromisoformat(x)
            return x

        start = _(start)
        end = _(end)

        if isinstance(start, datetime.date) and not isinstance(
            start, datetime.datetime
        ):
            start = datetime.datetime(start.year, start.month, start.day)

        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end = datetime.datetime(end.year, end.month, end.day)

        if end <= start:
            raise ValueError(f"End date {end} must be after start date {start}")

        increment = datetime.timedelta(hours=frequency)

        self.start = start
        self.end = end
        self.frequency = frequency

        date = start
        self.values = []
        while date <= end:
            self.values.append(date)
            date += increment

    def as_dict(self):
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "frequency": f"{self.frequency}h",
        }
