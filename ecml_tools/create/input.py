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
import logging
import math
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import cached_property

import numpy as np
from climetlab.core.order import build_remapping, normalize_order_by

from .config import Config
from .group import build_groups
from .template import substitute
from .utils import make_list_int, seconds, to_datetime, to_datetime_list

LOG = logging.getLogger(__name__)


def merge_remappings(*remappings):
    remapping = remappings[0]
    for other in remappings[1:]:
        if not other:
            continue
        assert other == remapping, (
            "Multiple inconsistent remappings not implemented",
            other,
            remapping,
        )
    return remapping


def assert_is_fieldset(obj):
    from climetlab.readers.grib.index import FieldSet

    assert isinstance(obj, FieldSet), type(obj)


def _get_data_request(data):
    date = None
    params_levels = defaultdict(set)
    params_steps = defaultdict(set)

    for field in data:
        if not hasattr(field, "as_mars"):
            continue
        if date is None:
            date = field.valid_datetime()
        if field.valid_datetime() != date:
            continue

        as_mars = field.as_mars()
        step = as_mars.get("step")
        levtype = as_mars.get("levtype", "sfc")
        param = as_mars["param"]
        levelist = as_mars.get("levelist", None)
        area = field.mars_area
        grid = field.mars_grid

        if levelist is None:
            params_levels[levtype].add(param)
        else:
            params_levels[levtype].add((param, levelist))

        if step:
            params_steps[levtype].add((param, step))

    def sort(old_dic):
        new_dic = {}
        for k, v in old_dic.items():
            new_dic[k] = sorted(list(v))
        return new_dic

    params_steps = sort(params_steps)
    params_levels = sort(params_levels)

    out = dict(param_level=params_levels, param_step=params_steps, area=area, grid=grid)
    return out


class Cache:
    pass


class Coords:
    def __init__(self, **dic):
        self._dic = dic
        self.shape = [len(v) for k, v in self._dic.items()]

        for k, v in self._dic.items():
            assert k in [
                "dates",
                "resolution",
                "grid_points",
                "ensembles",
                "variables",
                "grid_values",
            ], self._dic
            setattr(self, k, v)


class Input:
    _dates = None

    def __init__(self, config, selection=None, parent=None):
        assert parent != self, (parent, self)
        assert isinstance(config, dict), config
        self._config = config
        self.name = config.get("name")
        self.parent = parent
        self.cache = Cache()
        self._dates = self._build_dates(config, selection)

        self.order_by = parent.order_by
        self.flatten_grid = parent.flatten_grid

    def _build_dates(self, config, selection):
        if selection:
            assert len(selection) == 1, selection
            assert "dates" in selection, selection
            return deepcopy(selection["dates"])
        if "dates" in config:
            return deepcopy(config["dates"])
        return None

    def _build_coords(self):
        from_data = self.get_cube().user_coords
        from_config = self.order_by

        keys_from_config = list(from_config.keys())
        keys_from_data = list(from_data.keys())
        assert (
            keys_from_data == keys_from_config
        ), f"Critical error: {keys_from_data=} != {keys_from_config=}. {self=}"

        # assert keys[0] == "valid_datetime", keys
        # assert keys[1] == "param_level", keys
        # assert keys[2] == "number", keys

        variables_key = list(from_config.keys())[1]
        ensembles_key = list(from_config.keys())[2]
        self.cache.variables = from_data[variables_key]  # "param_level"
        self.cache.ensembles = from_data[ensembles_key]  # "number"

        if isinstance(from_config[variables_key], (list, tuple)):
            assert all(
                [
                    v == w
                    for v, w in zip(
                        from_data[variables_key], from_config[variables_key]
                    )
                ]
            ), (from_data[variables_key], from_config[variables_key])

        first_field = self.get_data[0]

        grid_points = first_field.grid_points()
        grid_values = list(range(len(grid_points[0])))

        self.cache.grid_points = grid_points
        self.cache.resolution = first_field.resolution
        self.cache.grid_values = grid_values

    def select(self, dates=None):
        if not dates:
            return self
        if not isinstance(dates, (list, tuple)):
            dates = [dates]
        return self.__class__(
            self._config, selection=dict(dates=dates), parent=self.parent
        )

    @property
    def variables(self):
        if not hasattr(self.cache, "variables"):
            self._build_coords()
        return self.cache.variables

    @property
    def ensembles(self):
        if not hasattr(self.cache, "ensembles"):
            self._build_coords()
        return self.cache.ensembles

    @property
    def resolution(self):
        if not hasattr(self.cache, "resolution"):
            self._build_coords()
        return self.cache.resolution

    @property
    def grid_values(self):
        if not hasattr(self.cache, "grid_values"):
            self._build_coords()
        return self.cache.grid_values

    @property
    def grid_points(self):
        if not hasattr(self.cache, "grid_points"):
            self._build_coords()
        return self.cache.grid_points

    @property
    def dates_obj(self):
        return build_groups(self.dates)

    @property
    def dates(self):
        return self._dates

    @property
    def remapping(self):
        return build_remapping(self._config.get("remapping", {}))

    def get_cube(self):
        ds = self.get_data
        for i, g in enumerate(ds):
            print(i, "field", g, self.dates, type(self))

        start = time.time()

        LOG.info("Sorting dataset %s %s", self.order_by, self.remapping)
        cube = ds.cube(
            self.order_by,
            remapping=self.remapping,
            flatten_values=self.flatten_grid,
            patches={"number": {None: 0}},
        )
        cube = cube.squeeze()
        LOG.info(f"Sorting done in {seconds(time.time()-start)}.")
        return cube

    @property
    def frequency(self):
        return self.dates_obj.frequency

    @property
    def shape(self):
        return [
            len(self.dates_obj.values),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        ]

    @property
    def coords(self):
        return {
            "dates": self.dates_obj.values,
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }

    def _raise_not_implemented(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def __repr__(self, *args, __indent__="\n", **kwargs):
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        if "dates" in self._config:
            more += " *dates-in-config "

        dates = str(self.dates)
        dates = dates[:5000]
        more += dates

        more = more[:5000]
        txt = f"{self.__class__.__name__}:{dates}{__indent__}{more}"
        if __indent__:
            txt = txt.replace("\n", "\n  ")
        return txt

    def check_compatibility(self, d1, d2):
        # These are the default checks
        # Derived classes should turn individual checks off if they are not needed
        # self.check_same_resolution(d1, d2)
        # self.check_same_frequency(d1, d2)
        # self.check_same_grid(d1, d2)
        # self.check_same_lengths(d1, d2)
        # self.check_same_variables(d1, d2)
        # self.check_same_dates(d1, d2)
        pass

    def check_same_dates(self, d1, d2):
        self.check_same_frequency(d1, d2)

        if d1.dates[0] != d2.dates[0]:
            raise ValueError(
                f"Incompatible start dates: {d1.dates[0]} and {d2.dates[0]} ({d1} {d2})"
            )

        if d1.dates[-1] != d2.dates[-1]:
            raise ValueError(
                f"Incompatible end dates: {d1.dates[-1]} and {d2.dates[-1]} ({d1} {d2})"
            )

    def check_same_frequency(self, d1, d2):
        if d1.frequency != d2.frequency:
            raise ValueError(
                f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})"
            )


class TerminalInput(Input):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        from climetlab import load_dataset, load_source

        self.func = {
            None: load_source,
            "load_source": load_source,
            "load_dataset": load_dataset,
            "climetlab.load_source": load_source,
            "climetlab.load_dataset": load_dataset,
            "cml.load_source": load_source,
            "cml.load_dataset": load_dataset,
        }[config.get("function")]

    @property
    def dates(self):
        # if self._dates is not None:
        #    return self._dates
        return self.parent.dates

    @property
    def get_data(self):
        assert self.dates is not None, self.dates
        config = deepcopy(self._config)

        args = config.get("args", [])
        kwargs = config.get("kwargs", {})

        vars = {}
        if self.dates:
            vars["dates"] = self.dates_obj.values

        previous = self.parent._find_previous(self)
        if previous:
            vars["previous"] = previous.get_data

        args = substitute(args, vars)
        kwargs = substitute(kwargs, vars)

        kwargs_str = ",".join([f"{k}={v}" for k, v in kwargs.items()])
        print(
            f"✅ get_data {id(self)}",
            args,
            kwargs_str,
            "DATES =",
            self.dates,
            len(self.dates_obj.values),
        )
        print("  for", self, "D=", self.dates)
        print(f"  {self.parent.parent.dates=}")
        ds = self.func(*args, **kwargs)
        print("get-data-len-ds", len(ds))
        return ds

    def __repr__(self):
        content = " ".join([f"{k}={v}" for k, v in self._config.items()])
        return f"{self.__class__.__name__}({self.name}) {content}"

    def get_data_request(self):
        ds = self.get_data
        return _get_data_request(ds)


class ConcatInput(Input):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inputs = config["concat"]
        self.inputs = []
        for i in inputs:
            i = build_input(i, parent=self, selection=kwargs.get("selection"))
            # todo: remove i if empty
            self.inputs.append(i)

    @property
    def dates(self):
        # if self._dates is not None:
        #    return self._dates
        assert len(self.inputs) == 1
        return self.inputs[0].dates
        # dates = []
        # for i in self.inputs:
        #     d = i.dates
        #     if d is None:
        #         continue
        #     dates += d
        # assert isinstance(dates[0], datetime.datetime), dates[0]
        # return sorted(dates)

    def __repr__(self):
        content = "\n".join([str(i) for i in self.inputs])
        dates = len(self.dates_obj.values)
        return str(dates) + "-" + super().__repr__(content)

    @property
    def get_data(self):
        if len(self.inputs) != 1:
            raise NotImplementedError()
        return self.inputs[0].get_data

    def get_cube(self):
        if len(self.inputs) != 1:
            raise NotImplementedError()
        return self.inputs[0].get_cube()


def merge_dicts(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict), (a, b)
        a = deepcopy(a)
        for k, v in b.items():
            if k not in a:
                a[k] = v
            else:
                a[k] = merge_dicts(a[k], v)
        return a

    return deepcopy(b)


class JoinInput(Input):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        inputs = config["join"]

        self.inputs = []
        previous_configs = {}
        for config in inputs:
            inherit = previous_configs.get(config.pop("inherit", None), {})
            config = merge_dicts(inherit, config)
            previous_configs[config["name"]] = config
            input = build_input(config, parent=self, selection=kwargs.get("selection"))
            self.inputs.append(input)

        for i in self.inputs[1:]:
            self.check_compatibility(self.inputs[0], i)

    def _find_previous(self, input):
        previous = None
        for i in self.inputs:
            if i == input:
                return previous
            previous = i
        return None

    def __repr__(self):
        content = "\n".join([str(i) for i in self.inputs])
        return f"{self.__class__.__name__}:\n{content}".replace("\n", "\n  ")

    @property
    def dates(self):
        return self._dates

    @property
    def remapping(self):
        remappings = [i.remapping for i in self.inputs]
        return merge_remappings(*remappings)

    @property
    def resolution(self):
        resolution = self.inputs[0].resolution
        for i in self.inputs[1:]:
            assert i.resolution == resolution
        return resolution

    @property
    def get_data(self):
        ds = self.inputs[0].get_data
        for i in self.inputs[1:]:
            ds += i.get_data
            assert_is_fieldset(ds), i
        return ds


def build_input(config, **kwargs):
    config = deepcopy(config)

    if isinstance(config, list):
        return ConcatInput(config, **kwargs)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    if "concat" in config:
        if not isinstance(config["concat"], list):
            raise ValueError(f"Concat must be a list, got {config['concat']}")
        return ConcatInput(config, **kwargs)

    if "join" in config:
        if not isinstance(config["join"], list):
            raise ValueError(f"Join must be a list, got {config['join']}")
        return JoinInput(config, **kwargs)

    return TerminalInput(config, **kwargs)
