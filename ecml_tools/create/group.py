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
            print(content)
            for g in self.groups:
                assert isinstance(g[0], datetime.datetime), g[0]
            print('val',self.values,self.n_groups)
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

        self._config=config
    
class ExpandGroups(Groups):

    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def values(self):
        return self._config.get("values")

class DateStartStopGroups(Groups):

    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def start(self):
        return self._get_date('start')
    
    @cached_property
    def end(self):
        return self._get_date('end')

    def _get_date(self,date_key):
        date = self._config[date_key]
        if type(date)==str:
            try:
                # Attempt to parse the date string with timestamp format
                check_timestamp=datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
                if check_timestamp:
                    return to_datetime(date)
            except ValueError:
                raise ValueError(f"{date_key}  must include timestamp not just date {date,type(date)}")
        elif type(date)==datetime.date:
            raise ValueError(f"{date_key}  must include timestamp not just date {date,type(date)}")
        else:
            return date

    def _validate_date_range(self):
        assert self.end >= self.start, "End date must be greater than or equal to start date."
        
    def _extract_frequency(self,frequency_str):
        freq_ending=frequency_str.lower()[-1]
        freq_mapping ={"h":int(frequency_str[:-1]),
                       "d":int(frequency_str[:-1])*24}
        try:
            return freq_mapping[freq_ending]
        except:
            raise ValueError(
                f"Frequency must be in hours or days (12h or 2d). {frequency_str}"
            )

    def _validate_frequency(self,freq,frequency_str):
        if freq > 24 and freq % 24 != 0:
            raise ValueError(
                f"Frequency must be less than 24h or a multiple of 24h. {frequency_str}"
            )
        
    @cached_property
    def step(self):
        _frequency_str = self._config.get("frequency", "24h")
        _freq = self._extract_frequency(_frequency_str)
        self._validate_frequency(_freq,_frequency_str)
        return datetime.timedelta(hours=_freq)

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
    def values(self):
        x = self.start
        dates = []
        while x <= self.end:
            dates.append(x)
           
            x += self.step
        assert isinstance(dates[0], datetime.datetime), dates[0]
        return dates

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


    @property
    def groups(self):
        # Return a list where each sublist contain the subgroups
        # of values according to the grouper_key
        return [Group(g) for _,g in itertools.groupby(self.values, key=self.grouper_key)]

    @cached_property
    def n_groups(self):
        return len(self.groups)



class EmptyGroups(BaseGroups):
    def __init__(self):
        self.values = []
        self.groups = []

    @property
    def frequency(self):
        return None


class GroupsIntersection(BaseGroups):
    def __init__(self, a, b):
        assert isinstance(a,Groups), a 
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
        list: lambda x: ExpandGroups(dict(values=x)), #Expand
        Group: lambda x: ExpandGroups(dict(values=x)), #Expand
        DictObj: lambda x: DateStartStopGroups(dict(x["dates"])) if "dates" in x and len(x) == 1 else DateStartStopGroups(x), #StarStoptEnd
        dict: lambda x: DateStartStopGroups(dict(x["dates"])) if "dates" in x and len(x) == 1 else DateStartStopGroups(x), #StarStoptEnd
    }

    return type_actions[type(obj)](obj)