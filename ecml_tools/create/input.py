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
from collections import defaultdict
from copy import deepcopy
from functools import cached_property

import numpy as np

from .config import Config
from .loops import expand_loops
from .template import substitute
from .utils import make_list_int, seconds, to_datetime, to_datetime_list

LOG = logging.getLogger(__name__)


def assert_is_fieldset(obj):
    from climetlab.readers.grib.index import FieldSet

    assert isinstance(obj, FieldSet), type(obj)


class SourceTemplates:
    def __init__(self, *args):
        assert isinstance(args[0], (dict, SourceTemplate)), args[0]
        self._elements = [
            c if isinstance(c, SourceTemplate) else SourceTemplate(c) for c in args
        ]

    def __iter__(self):
        for i in self._elements:
            yield i

    def filter(self, func):
        return SourceTemplates(*[i for i in self if func(i)])

    def __repr__(self) -> str:
        return "\n".join(str(i) for i in self)


class SourceTemplate:
    _inheritance_done = False
    _inheritance_others = None

    def __init__(self, dic):
        assert isinstance(dic, dict), dic
        assert len(dic) == 1, dic
        name = list(dic.keys())[0]
        config = dic[name]

        self.name = name
        self.kwargs = config.get("kwargs", {})
        self.inherit = config.get("inherit", None)
        self.function = config.get("function", None)

        assert self.inherit is None or isinstance(self.inherit, str), self.inherit

        if self.kwargs["name"] in ["forcing", "constants"]:
            # add $ to source_or_dataset for constants source.
            # TODO: refactor to remove this.
            v = self.kwargs.get("source_or_dataset")
            if isinstance(v, str) and not v.startswith("$"):
                self.kwargs["source_or_dataset"] = "$" + v

    def get_first_field(self):
        return self.do_load()[0]

    def process_inheritance(self, others):
        for o in others:
            if o == self:
                continue
            name = o.name
            if name.startswith("$"):
                name = name[1:]
            if name != self.inherit:
                continue
            if not o._inheritance_done:
                o.process_inheritance(others)

            kwargs = {}
            kwargs.update(o.kwargs)
            kwargs.update(self.kwargs)  # self.kwargs has priority
            self.kwargs = kwargs

        self._inheritance_others = others
        self._inheritance_done = True

    def instanciate(self, *args, **kwargs):
        new_kwargs = substitute(self.kwargs.copy(), *args, **kwargs)
        return WrappedSource(
            name=self.name,
            kwargs=new_kwargs,
            function=self.function,
        )

    def __repr__(self) -> str:
        def repr(v):
            if isinstance(v, list):
                return f"{'/'.join(str(x) for x in v)}"
            return str(v)

        details = ", ".join(f"{k}={repr(v)}" for k, v in self.kwargs.items())
        return f"SourceTemplate({self.name}, {details})<{self.inherit}"


class WrappedSource:
    _do_load = None

    def __init__(self, name, kwargs, function=None):
        self.name = name  # for logging purposes only
        self.kwargs = kwargs
        self.function = function

    def get_datetimes(self):
        # parse the kwargs to get the datetimes
        name = self.kwargs.get("name", None)

        assert name in [
            "era5-accumulations",
            "constants",
            "mars",
        ], f"{name} not implemented"

        if name == "constants":
            return None

        if name == "era5-accumulations":
            return None

        if name == "mars":
            is_hindast = "hdate" in self.kwargs

            date = self.kwargs.get("date", [])
            hdate = self.kwargs.get("hdate", [])
            time = self.kwargs.get("time", [0])
            step = self.kwargs.get("step", [0])

            date = to_datetime_list(date)
            hdate = to_datetime_list(hdate)
            time = make_list_int(time)
            step = make_list_int(step)

            assert isinstance(date, (list, tuple)), date
            assert isinstance(time, (list, tuple)), time
            assert isinstance(step, (list, tuple)), step

            if is_hindast:
                assert isinstance(hdate, (list, tuple)), hdate
                if len(date) > 1 and len(hdate) > 1:
                    raise NotImplementedError(
                        (
                            f"Cannot have multiple dates in {self} "
                            "when using hindcast {date=}, {hdate=}"
                        )
                    )
                date = hdate
                del hdate

            if len(step) > 1 and len(time) > 1:
                raise NotImplementedError(
                    f"Cannot have multiple steps and multiple times in {self}"
                )

            datetimes = set()
            for d, t, s in itertools.product(date, time, step):
                new = build_datetime(date=d, time=t, step=s)
                if new in datetimes:
                    raise DuplicateDateTimeError(
                        f"Duplicate datetime '{new}' when processing << {self} >> already in {datetimes}"
                    )
                datetimes.add(new)
            return sorted(list(datetimes))

        raise ValueError(f"{name=} Cannot count number of elements in {self}")

    def do_load(self, partial, others=None):
        if not self._do_load or self._do_load[1] != partial:
            from climetlab import load_dataset, load_source

            func = {
                None: load_source,
                "load_source": load_source,
                "load_dataset": load_dataset,
                "climetlab.load_source": load_source,
                "climetlab.load_dataset": load_dataset,
                "cml.load_source": load_source,
                "cml.load_dataset": load_dataset,
            }[self.function]

            kwargs = dict(**self.kwargs)
            if others:
                kwargs = substitute(self.kwargs.copy(), vars=others)

            if partial:
                if "date" in kwargs and isinstance(kwargs["date"], list):
                    kwargs["date"] = [kwargs["date"][0]]

            LOG.info(f"Loading {self.name} with {func} {kwargs}")
            ds = func(**kwargs)

            LOG.info(f"  Loading {self.name} of len {len(ds)}: {ds}")
            self._do_load = (ds, partial)
        return self._do_load[0]


def build_datetime(date, time, step):
    """
    Build a datetime object from a date, time, and step.

    Args:
        date (str or datetime.datetime): The date to use. If a string, it will be converted to a datetime object.
        time (int or str): The time to use. If an integer, it will be converted to a string in the format HHMM or HH.
        If a string, it must be in the format HHMM.
        step (int): The number of hours to add to the resulting datetime object.

    Returns:
        datetime.datetime: The resulting datetime object.

    Examples:
        >>> build_datetime('2022-01-01', 12, 0)
        datetime.datetime(2022, 1, 1, 12, 0)
        >>> build_datetime('2022-01-01', '12', 0)
        Traceback (most recent call last):
         ...
        AssertionError: 12
        >>> build_datetime('2022-01-01', '1200', 0)
        datetime.datetime(2022, 1, 1, 12, 0)
        >>> build_datetime(datetime.datetime(2022, 1, 1), '1230', 0)
        datetime.datetime(2022, 1, 1, 12, 30)
        >>> build_datetime('2022-01-01', 12, 2)
        datetime.datetime(2022, 1, 1, 14, 0)
    """
    if isinstance(date, str):
        date = to_datetime(date)

    if isinstance(time, int):
        if time < 24:
            time = f"{time:02d}00"
        else:
            time = f"{time:04d}"

    assert isinstance(date, datetime.datetime), date
    assert date.hour == 0 and date.minute == 0 and date.second == 0, date

    assert isinstance(time, str), time
    assert len(time) == 4, time
    assert int(time) >= 0 and int(time) < 2400, time
    if 0 < int(time) < 100:
        LOG.warning(f"{time=}, using time with minutes is unusual.")

    dt = datetime.datetime(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=int(time[0:2]),
        minute=int(time[2:4]),
    )

    if step:
        dt += datetime.timedelta(hours=step)

    return dt


class Loops:
    def __init__(self, main_config, parent):
        # instanciated by subclasses
        assert self.partial in [True, False], self.partial

        assert isinstance(main_config, Config), main_config
        self.parent = parent

        loops_config = main_config.loop
        if not loops_config:
            raise NotImplementedError("No loop")

        all_templates = [SourceTemplate(c) for c in main_config.input]

        self._elements = []
        for loop in loops_config:
            loop = Loop(loop, all_templates, parent=self, partial=self.partial)
            self._elements.append(loop)

    @cached_property
    def n_iter_loops(self):
        return sum([loop.n_iter_loops for loop in self])

    def _chunking(self, config):
        # to be called by OutputSpecs
        # we may want to check each loop.
        for loop in self:
            for inputs in loop.iterate_cubes():
                cube = inputs.to_cube()
                return cube.chunking(config)

    def __iter__(self):
        for i in self._elements:
            yield i

    def iter_cubes(self):
        for loop in self:
            yield from loop.iterate_cubes()

    @cached_property
    def n_cubes(self):
        n = 0
        for loop in self:
            for i in loop.iterate_cubes():
                n += 1
        return n

    @cached_property
    def _datetimes_and_frequency(self):
        # merge datetimes from all loops and check there are no duplicates
        datetimes = set()
        for i in self:
            assert isinstance(i, Loop), i
            new = i.get_datetimes()
            for d in new:
                assert d not in datetimes, (d, datetimes)
                datetimes.add(d)
        datetimes = sorted(list(datetimes))

        def check(datetimes):
            if not datetimes:
                raise ValueError("No datetimes found.")
            if len(datetimes) == 1:
                raise ValueError("Only one datetime found.")

            delta = None
            for i in range(1, len(datetimes)):
                new = (datetimes[i] - datetimes[i - 1]).total_seconds() / 3600
                if not delta:
                    delta = new
                    continue
                if new != delta:
                    raise ValueError(
                        f"Datetimes are not regularly spaced: "
                        f"delta={new} hours  (date {i-1}={datetimes[i-1]}  date {i}={datetimes[i]}) "
                        f"Expecting {delta} hours  (date {0}={datetimes[0]}  date {1}={datetimes[1]}) "
                    )

        check(datetimes)

        freq = (datetimes[1] - datetimes[0]).total_seconds() / 3600
        assert round(freq) == freq, freq
        assert int(freq) == freq, freq
        frequency = int(freq)

        return datetimes, frequency

    @property
    def frequency(self):
        return self._datetimes_and_frequency[1]

    def get_datetimes(self):
        return self._datetimes_and_frequency[0]

    @cached_property
    def data_description(self):
        infos = []
        for loop in self:
            first = loop.first.data_description
            coords = deepcopy(first.coords)
            assert (
                "valid_datetime" in coords
            ), f"valid_datetime not found in coords {coords}"
            coords["valid_datetime"] = loop.get_datetimes()

            loop_data_description = DataDescription(
                first_field=first.first_field,
                grid_points=first.grid_points,
                resolution=first.resolution,
                variables=first.variables,
                data_request=first.data_request,
                coords=coords,
            )
            infos.append(loop_data_description)

        # check all are the same
        ref = infos[0]
        for c in infos:
            assert (np.array(ref.grid_points) == np.array(c.grid_points)).all(), (
                "grid_points mismatch",
                c.grid_points,
                ref.grid_points,
                type(ref.grid_points),
            )
            assert ref.resolution == c.resolution, (
                "resolution mismatch",
                c.resolution,
                ref.resolution,
            )
            assert ref.variables == c.variables, (
                "variables mismatch",
                c.variables,
                ref.variables,
            )

        coords = deepcopy(ref.coords)
        assert (
            "valid_datetime" in coords
        ), f"valid_datetime not found in coords {coords}"
        coords["valid_datetime"] = self.get_datetimes()

        for info in infos:
            for name, values in info.coords.items():
                if name == "valid_datetime":
                    continue
                assert values == ref.coords[name], (values, ref.coords[name])

        return DataDescription(
            ref.first_field,
            ref.grid_points,
            ref.resolution,
            coords,
            ref.variables,
            ref.data_request,
        )

    @property
    def first_field(self):
        return self.data_description.first_field

    @property
    def grid_points(self):
        return self.data_description.grid_points

    @property
    def resolution(self):
        return self.data_description.resolution

    @property
    def data_request(self):
        return self.data_description.data_request

    @property
    def coords(self):
        return self.data_description.coords

    @property
    def variables(self):
        return self.data_description.variables


class FullLoops(Loops):
    partial = False


class PartialLoops(Loops):
    partial = True


class Loop:
    def __init__(self, config, all_templates, parent, partial):
        assert isinstance(config, dict), config
        assert len(config) == 1, config

        self.parent = parent
        self.name = list(config.keys())[0]
        self.config = deepcopy(config[self.name])
        self.partial = partial

        # applied to all if unspecified
        self.applies_to = self.config.get("applies_to", [i.name for i in all_templates])

        self.templates = [t for t in all_templates if t.name in self.applies_to]
        for t in self.templates:
            t.process_inheritance(self.templates)

        self.values = {}
        for k, v in self.config.items():
            self.values[k] = expand_loops(v)

    def __repr__(self) -> str:
        def repr_lengths(v):
            return f"{','.join([str(len(x)) for x in v])}"

        lenghts = [f"{k}({repr_lengths(v)})" for k, v in self.values.items()]
        return f"Loop({self.name}, {','.join(lenghts)}) {self.config}"

    @cached_property
    def n_iter_loops(self):
        return len(list(itertools.product(*self.values.values())))

    def iterate_cubes(self):
        for values in itertools.product(*self.values.values()):
            vars = {k: v for k, v in zip(self.values.keys(), values)}
            assert isinstance(vars, dict), (vars, self.values)
            instanciated_inputs = [
                t.instanciate(vars=vars, ignore_missing=True) for t in self.templates
            ]
            yield Inputs(
                inputs=instanciated_inputs,
                loop_config=self.config,
                output_specs=self.parent.parent.output_specs,
                partial=self.partial,
            )

    @property
    def first(self):
        vars = {k: lst[0] for k, lst in self.values.items() if lst}

        instanciated_inputs = [
            t.instanciate(vars=vars, ignore_missing=True) for t in self.templates
        ]

        return Inputs(
            inputs=instanciated_inputs,
            loop_config=self.config,
            output_specs=self.parent.parent.output_specs,
            partial=self.partial,
        )

    def get_datetimes(self):
        # merge datetimes from all lazycubes and check there are no duplicates
        datetimes = set()

        for i in self.iterate_cubes():
            assert isinstance(i, Inputs), i
            new = i.get_datetimes()

            duplicates = datetimes.intersection(set(new))
            if duplicates:
                raise DuplicateDateTimeError(
                    f"{len(duplicates)} duplicated datetimes "
                    f"'{sorted(list(duplicates))[0]},...' when processing << {self} >>"
                )

            datetimes = datetimes.union(set(new))
        return sorted(list(datetimes))


class DuplicateDateTimeError(ValueError):
    pass


class Inputs:
    _do_load = None

    def __init__(self, inputs, loop_config, output_specs, partial):
        assert isinstance(inputs, list), inputs
        assert isinstance(inputs[0], WrappedSource), inputs[0]

        self._loop_config = loop_config
        self.inputs = inputs
        self.output_specs = output_specs
        self.partial = partial

    def __iter__(self):
        for i in self.inputs:
            yield i

    @property
    def length(self):
        return 1

    def __repr__(self) -> str:
        out = f"Inputs ({self.length}):\n"
        out += f" loop_config: {self._loop_config}"
        out += " Inputs:\n"
        for i in self.inputs:
            out += f"  {i}\n"
        return out

    def do_load(self):
        partial = self.partial
        if self._do_load is None or self._do_load[1] != partial:
            datasets = {}
            for i in self:
                ds = i.do_load(partial=partial, others=datasets)
                datasets[i.name] = ds

            out = None
            for ds in datasets.values():
                if out is None:
                    out = ds
                else:
                    out += ds

            assert_is_fieldset(out)
            self._do_load = (out, partial)

        return self._do_load[0]

    def get_datetimes(self):
        # get datetime from each wrapped source
        # and make sure they are the same or None
        datetimes = None
        previous_name = None
        for i in self:
            new = i.get_datetimes()
            if new is None:
                continue
            new = sorted(list(new))
            if datetimes is None:
                datetimes = new

            if datetimes != new:
                raise ValueError(
                    "Mismatch in datetimes", previous_name, datetimes, i.name, new
                )
            previous_name = i.name

        if datetimes is None:
            raise ValueError(f"No datetimes found in {self}")

        return datetimes

    def to_cube(self):
        cube, _ = self._to_data_and_cube()
        return cube

    def _to_data_and_cube(self):
        data = self.do_load()

        start = time.time()
        LOG.info(
            "Sorting dataset %s %s",
            self.output_specs.order_by,
            self.output_specs.remapping,
        )
        cube = data.cube(
            self.output_specs.order_by,
            remapping=self.output_specs.remapping,
            flatten_values=self.output_specs.flatten_grid,
            patches={"number": {None: 0}},
        )
        cube = cube.squeeze()
        LOG.info(f"Sorting done in {seconds(time.time()-start)}.")

        def check(actual_dic, requested_dic):
            assert self.output_specs.statistics in actual_dic

            for key in set(list(actual_dic.keys()) + list(requested_dic.keys())):
                actual = actual_dic[key]
                requested = requested_dic[key]

                actual = list(actual)

                if requested == "ascending":
                    assert actual == sorted(
                        actual
                    ), f"Requested= {requested} Actual= {actual}"
                    continue
                assert actual == requested, f"Requested= {requested} Actual= {actual}"

        check(actual_dic=cube.user_coords, requested_dic=self.output_specs.order_by)

        return cube, data

    @property
    def data_description(self):
        cube, data = self._to_data_and_cube()

        first_field = data[0]
        data_request = self._get_data_request(data)
        grid_points = first_field.grid_points()
        resolution = first_field.resolution
        coords = cube.user_coords
        variables = list(coords[list(coords.keys())[1]])

        return DataDescription(
            first_field, grid_points, resolution, coords, variables, data_request
        )

    def _get_data_request(self, data):
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

        out = dict(
            param_level=params_levels, param_step=params_steps, area=area, grid=grid
        )
        return out


def _format_list(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], datetime.datetime):
            is_regular = True
            delta = x[1] - x[0]
            for prev, current in zip(x[:-1], x[1:]):
                if current - prev != delta:
                    is_regular = False
                    break
            if is_regular:
                return f"{_format_list(x[0])}/to/{_format_list(x[-1])}/by/{delta.total_seconds()/3600}"

        txt = "/".join(_format_list(_) for _ in x)
        if len(txt) > 200:
            txt = txt[:50] + "..." + txt[-50:]
        return txt

    if isinstance(x, datetime.datetime):
        return x.strftime("%Y-%m-%d.%H:%M")
    return str(x)


def check_data_specs(
    first_field, grid_points, resolution, coords, variables, data_request
):
    assert len(set(variables)) == len(variables), (
        "Duplicate variables",
        variables,
    )

    assert grid_points[0].shape == grid_points[1].shape, (
        grid_points[0].shape,
        grid_points[1].shape,
        grid_points[0],
        grid_points[1],
    )

    assert len(grid_points) == 2, grid_points

    expected = math.prod(first_field.shape)
    assert len(grid_points[0]) == expected, (len(grid_points[0]), expected)


class DataDescription:
    def __init__(
        self, first_field, grid_points, resolution, coords, variables, data_request
    ):
        check_data_specs(
            first_field, grid_points, resolution, coords, variables, data_request
        )

        self.first_field = first_field
        self.grid_points = grid_points
        self.resolution = resolution
        self.coords = coords
        self.variables = variables
        self.data_request = data_request

    def __repr__(self):
        shape = (
            f"{','.join([str(len(v)) for k, v in self.coords.items()])},"
            + f"{','.join([str(_) for _ in self.first_field.shape])}"
        )
        shape = shape.rjust(20)
        return (
            f"DataDescription(first_field={self.first_field}, "
            f"resolution={self.resolution}, "
            f"variables={'/'.join(self.variables)})"
            f" coords={', '.join([k + ':' + _format_list(v) for k, v in self.coords.items()])}"
            f" {shape}"
        )
