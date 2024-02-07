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
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a dict. {config}")        
        # assert it is not a nested dictionary
        for k, v in config.items():
            if isinstance(v, dict):
                raise ValueError(f"Values can't be a dictionary. {k,v}")
        self._config = config

    @property
    def values(self):
        raise NotImplementedError()

    @property
    def groups(self):
        raise NotImplementedError(type(self))

    @staticmethod
    def conditions():
        raise NotImplementedError()

class ValuesExpand(Expand):
    def __init__(self, config):
        super().__init__(config)
        print('ValuesExpand BASE CONFIG',self._config)
        if not isinstance(self._config["values"], list):
            raise ValueError(f"Values must be a list. {self._config}")

    @property
    def values(self):
        return self._config["values"]
    
    @staticmethod
    def conditions(config):
        return [isinstance(config.get("values"), list) and len(config) == 1]

class DateStartStopExpand(Expand):
    def __init__(self, config):
        super().__init__(config)
        if "stop" in self._config:
            raise ValueError(f"Use 'end' not 'stop' in loop. {self._config}")

        print('DateStartStopExpand BASE CONFIG',self._config)
        self.start = self._get_date('start')
        self.end = self._get_date('end')

        self._validate_date_range()

        _frequency_str = self._config.get("frequency", "24h")
        _freq = self._extract_frequency(_frequency_str)
        self._validate_frequency(_freq,_frequency_str)
        self.step = datetime.timedelta(hours=_freq)

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

    @property
    def values(self): 
        x = self.start
        all = []
        while x <= self.end:
            all.append(x)
            yield x
            x += self.step

    @cached_property
    def groups(self):
        groups = []
        for _, g in itertools.groupby(self.values, key=self.grouper_key):
            g = [self.format(x) for x in g]
            groups.append(g)
        return groups

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
        
    @staticmethod
    def conditions(config):
        return [
            config.get("group_by") in ["monthly","daily","weekly",],
            isinstance(config.get("start"), datetime.datetime),
            isinstance(config.get("end"), datetime.datetime),
            "frequency" in config,
            config.get("kind") == "dates" and "start" in config]

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



def expand_class(config):
    base_dict_builder = {
        list: lambda x:  {"values": x},
        dict: lambda x: x,
    }
    try: 
        base_dict=base_dict_builder[type(config)](config)
        if "start" not in config and "values" not in config:
            raise ValueError(f"Cannot expand loop from {config}")
        else:

            candidates_classes={ValuesExpand:ValuesExpand.conditions(base_dict),
                                DateStartStopExpand: DateStartStopExpand.conditions(base_dict)}
            for expanded_class, list_of_conditions in candidates_classes.items():
                if any(list_of_conditions):
                    return expanded_class
    except:
        raise ValueError(f"Cannot expand loop from {config}")