# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import importlib
import logging
import textwrap
import time
from collections import defaultdict
from copy import deepcopy
from functools import cached_property, wraps

import numpy as np
from climetlab.core.order import build_remapping

from .group import build_groups
from .template import resolve, substitute
from .utils import seconds

LOG = logging.getLogger(__name__)


def import_function(name, kind):
    return importlib.import_module(
        f"..functions.{kind}.{name}",
        package=__name__,
    ).execute


def is_function(name, kind):
    try:
        import_function(name, kind)
        return True
    except ImportError:
        return False


def assert_is_fieldset(obj):
    from climetlab.indexing.fieldset import FieldSet

    assert isinstance(obj, FieldSet), type(obj)


def _datasource_request(data):
    date = None
    params_levels = defaultdict(set)
    params_steps = defaultdict(set)

    area = grid = None

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

    return dict(
        param_level=params_levels, param_step=params_steps, area=area, grid=grid
    )


class Coords:
    def __init__(self, owner):
        self.owner = owner

    @cached_property
    def _build_coords(self):
        from_data = self.owner.get_cube().user_coords
        from_config = self.owner.context.order_by

        keys_from_config = list(from_config.keys())
        keys_from_data = list(from_data.keys())
        assert (
            keys_from_data == keys_from_config
        ), f"Critical error: {keys_from_data=} != {keys_from_config=}. {self.owner=}"

        variables_key = list(from_config.keys())[1]
        ensembles_key = list(from_config.keys())[2]

        if isinstance(from_config[variables_key], (list, tuple)):
            assert all(
                [
                    v == w
                    for v, w in zip(
                        from_data[variables_key], from_config[variables_key]
                    )
                ]
            ), (
                from_data[variables_key],
                from_config[variables_key],
            )

        self._variables = from_data[variables_key]  # "param_level"
        self._ensembles = from_data[ensembles_key]  # "number"

        first_field = self.owner.datasource[0]
        grid_points = first_field.grid_points()

        lats, lons = grid_points
        north = np.amax(lats)
        south = np.amin(lats)
        east = np.amax(lons)
        west = np.amin(lons)

        assert -90 <= south <= north <= 90, (south, north, first_field)
        assert (-180 <= west <= east <= 180) or (0 <= west <= east <= 360), (
            west,
            east,
            first_field,
        )

        grid_values = list(range(len(grid_points[0])))

        self._grid_points = grid_points
        self._resolution = first_field.resolution
        self._grid_values = grid_values

    @cached_property
    def variables(self):
        self._build_coords
        return self._variables

    @cached_property
    def ensembles(self):
        self._build_coords
        return self._ensembles

    @cached_property
    def resolution(self):
        self._build_coords
        return self._resolution

    @cached_property
    def grid_values(self):
        self._build_coords
        return self._grid_values

    @cached_property
    def grid_points(self):
        self._build_coords
        return self._grid_points


class HasCoordsMixin:
    @cached_property
    def variables(self):
        return self._coords.variables

    @cached_property
    def ensembles(self):
        return self._coords.ensembles

    @cached_property
    def resolution(self):
        return self._coords.resolution

    @cached_property
    def grid_values(self):
        return self._coords.grid_values

    @cached_property
    def grid_points(self):
        return self._coords.grid_points

    @cached_property
    def dates(self):
        if self._dates is None:
            raise ValueError(f"No dates for {self}")
        return self._dates.values

    @cached_property
    def frequency(self):
        return self._dates.frequency

    @cached_property
    def shape(self):
        return [
            len(self.dates),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        ]

    @cached_property
    def coords(self):
        return {
            "dates": self.dates,
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }


class Action:
    def __init__(self, context, action_path, /, *args, **kwargs):
        if "args" in kwargs and "kwargs" in kwargs:
            """We have:
               args = []
               kwargs = {args: [...], kwargs: {...}}
            move the content of kwargs to args and kwargs.
            """
            assert len(kwargs) == 2, (args, kwargs)
            assert not args, (args, kwargs)
            args = kwargs.pop("args")
            kwargs = kwargs.pop("kwargs")

        assert isinstance(context, Context), type(context)
        self.context = context
        self.kwargs = kwargs
        self.args = args
        self.action_path = action_path

    @classmethod
    def _short_str(cls, x):
        x = str(x)
        if len(x) < 1000:
            return x
        return x[:1000] + "..."

    def __repr__(self, *args, _indent_="\n", _inline_="", **kwargs):
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        more = more[:5000]
        txt = f"{self.__class__.__name__}: {_inline_}{_indent_}{more}"
        if _indent_:
            txt = txt.replace("\n", "\n  ")
        return txt

    def select(self, dates, **kwargs):
        self._raise_not_implemented()

    def _raise_not_implemented(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def _trace_select(self, dates):
        return f"{self.__class__.__name__}({shorten(dates)})"


def shorten(dates):
    if isinstance(dates, (list, tuple)):
        dates = [d.isoformat() for d in dates]
        if len(dates) > 5:
            return f"{dates[0]}...{dates[-1]}"
    return dates


def check_references(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.context.notify_result(self.action_path, result)
        return result

    return wrapper


TRACE_INDENT = 0


def step(action_path):
    return f"[{'.'.join(action_path)}]"


def trace(emoji, *args):
    print(emoji, " " * TRACE_INDENT, *args)


def trace_datasource(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        global TRACE_INDENT
        trace(
            "🌍",
            "=>",
            step(self.action_path),
            self._trace_datasource(*args, **kwargs),
        )
        TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        TRACE_INDENT -= 1
        trace(
            "🍎",
            "<=",
            step(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper


def trace_select(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        global TRACE_INDENT
        trace(
            "👓",
            "=>",
            ".".join(self.action_path),
            self._trace_select(*args, **kwargs),
        )
        TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        TRACE_INDENT -= 1
        trace(
            "🍍",
            "<=",
            ".".join(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper


class Result(HasCoordsMixin):
    empty = False

    def __init__(self, context, action_path, dates):
        assert isinstance(context, Context), type(context)

        assert action_path is None or isinstance(action_path, list), action_path
        action_path = tuple(action_path or [])

        self.context = context
        self._coords = Coords(self)
        self._dates = dates
        self.action_path = action_path
        if action_path is not None:
            context.register_reference(action_path, self)

    @property
    @trace_datasource
    def datasource(self):
        self._raise_not_implemented()

    @property
    def data_request(self):
        """Returns a dictionary with the parameters needed to retrieve the data."""
        return _datasource_request(self.datasource)

    def get_cube(self):
        trace("🧊", f"getting cube from {self.__class__.__name__}")
        ds = self.datasource
        assert_is_fieldset(ds)

        remapping = self.context.remapping
        order_by = self.context.order_by
        flatten_grid = self.context.flatten_grid
        start = time.time()
        LOG.info("Sorting dataset %s %s", order_by, remapping)
        assert order_by, order_by
        cube = ds.cube(
            order_by,
            remapping=remapping,
            flatten_values=flatten_grid,
            patches={"number": {None: 0}},
        )
        cube = cube.squeeze()
        LOG.info(f"Sorting done in {seconds(time.time()-start)}.")

        return cube

    def __repr__(self, *args, _indent_="\n", **kwargs):
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        dates = " no-dates"
        if self._dates is not None:
            dates = f" {len(self.dates)} dates"
            dates += " ("
            dates += "/".join(d.strftime("%Y-%m-%d:%H") for d in self.dates)
            if len(dates) > 100:
                dates = dates[:100] + "..."
            dates += ")"

        more = more[:5000]
        txt = f"{self.__class__.__name__}:{dates}{_indent_}{more}"
        if _indent_:
            txt = txt.replace("\n", "\n  ")
        return txt

    def _raise_not_implemented(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.__class__.__name__}({shorten(self.dates)})"


class EmptyResult(Result):
    empty = True

    @cached_property
    @trace_datasource
    def datasource(self):
        from climetlab import load_source

        return load_source("empty")

    @property
    def variables(self):
        return []


class FunctionResult(Result):
    def __init__(self, context, action_path, dates, action):
        super().__init__(context, action_path, dates)
        assert isinstance(action, Action), type(action)
        self.action = action

        self.args, self.kwargs = substitute(
            context, (self.action.args, self.action.kwargs)
        )

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({shorten(self.dates)})"

    @cached_property
    @check_references
    @trace_datasource
    def datasource(self):
        args, kwargs = resolve(self.context, (self.args, self.kwargs))

        try:
            return self.action.function(
                FunctionContext(self), self.dates, *args, **kwargs
            )
        except Exception:
            LOG.error(f"Error in {self.action.function.__name__}", exc_info=True)
            raise

    def __repr__(self):
        try:
            return f"{self.action.name}({shorten(self.dates)})"
        except Exception:
            return f"{self.__class__.__name__}(unitialised)"

    @property
    def function(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class JoinResult(Result):
    def __init__(self, context, action_path, dates, results, **kwargs):
        super().__init__(context, action_path, dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @check_references
    @trace_datasource
    def datasource(self):
        ds = EmptyResult(self.context, None, self._dates).datasource
        for i in self.results:
            ds += i.datasource
            assert_is_fieldset(ds), i
        return ds

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class FunctionAction(Action):
    def __init__(self, context, action_path, _name, **kwargs):
        super().__init__(context, action_path, **kwargs)
        self.name = _name

    @trace_select
    def select(self, dates):
        return FunctionResult(self.context, self.action_path, dates, action=self)

    @property
    def function(self):
        return import_function(self.name, "actions")

    def __repr__(self):
        content = ""
        content += ",".join([self._short_str(a) for a in self.args])
        content += " ".join(
            [self._short_str(f"{k}={v}") for k, v in self.kwargs.items()]
        )
        content = self._short_str(content)
        return super().__repr__(_inline_=content, _indent_=" ")

    def _trace_select(self, dates):
        return f"{self.name}({shorten(dates)})"


class ConcatResult(Result):
    def __init__(self, context, action_path, results):
        super().__init__(context, action_path, dates=None)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @check_references
    @trace_datasource
    def datasource(self):
        ds = EmptyResult(self.context, None, self.dates).datasource
        for i in self.results:
            ds += i.datasource
            assert_is_fieldset(ds), i
        return ds

    @property
    def variables(self):
        """Check that all the results objects have the same variables."""
        variables = None
        for f in self.results:
            if f.empty:
                continue
            if variables is None:
                variables = f.variables
            assert variables == f.variables, (variables, f.variables)
        assert variables is not None, self.results
        return variables

    @property
    def dates(self):
        """Merge the dates of all the results objects."""
        dates = []
        for i in self.results:
            d = i.dates
            if d is None:
                continue
            dates += d
        assert isinstance(dates[0], datetime.datetime), dates[0]
        return sorted(dates)

    @property
    def frequency(self):
        return build_groups(self.dates).frequency

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class ActionWithList(Action):
    result_class = None

    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        self.actions = [
            action_factory(c, context, action_path + [str(i)])
            for i, c in enumerate(configs)
        ]

    def __repr__(self):
        content = "\n".join([str(i) for i in self.actions])
        return super().__repr__(content)


class PipeAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        assert len(configs) > 1, configs
        current = action_factory(configs[0], context, action_path + ["0"])
        for i, c in enumerate(configs[1:]):
            current = step_factory(
                c, context, action_path + [str(i + 1)], previous_step=current
            )
        self.last_step = current

    @trace_select
    def select(self, dates):
        return self.last_step.select(dates)

    def __repr__(self):
        return super().__repr__(self.last_step)


class StepResult(Result):
    def __init__(self, context, action_path, dates, action, upstream_result):
        super().__init__(context, action_path, dates)
        assert isinstance(upstream_result, Result), type(upstream_result)
        self.upstream_result = upstream_result
        self.action = action

    @property
    @check_references
    @trace_datasource
    def datasource(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")
        # return self.upstream_result.datasource


class StepAction(Action):
    result_class = None

    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, *args, **kwargs)
        self.previous_step = previous_step

    @trace_select
    def select(self, dates):
        return self.result_class(
            self.context,
            self.action_path,
            dates,
            self,
            self.previous_step.select(dates),
        )

    def __repr__(self):
        return super().__repr__(self.previous_step, _inline_=str(self.kwargs))


class StepFunctionResult(StepResult):
    @cached_property
    @check_references
    @trace_datasource
    def datasource(self):
        try:
            return self.action.function(
                FunctionContext(self),
                self.upstream_result.datasource,
                **self.action.kwargs,
            )

        except Exception:
            LOG.error(f"Error in {self.action.name}", exc_info=True)
            raise

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({shorten(self.dates)})"


class FilterStepResult(StepResult):
    @property
    @check_references
    @trace_datasource
    def datasource(self):
        ds = self.content.datasource
        assert_is_fieldset(ds)
        ds = ds.sel(**self.action.kwargs)
        assert_is_fieldset(ds)
        return ds


class FilterStepAction(StepAction):
    result_class = FilterStepResult


class FunctionStepAction(StepAction):
    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, previous_step, *args, **kwargs)
        self.name = args[0]
        self.function = import_function(self.name, "steps")

    result_class = StepFunctionResult


class ConcatAction(ActionWithList):
    @trace_select
    def select(self, dates):
        return ConcatResult(self.context, [a.select(dates) for a in self.actions])


class JoinAction(ActionWithList):
    @trace_select
    def select(self, dates):
        return JoinResult(
            self.context,
            self.action_path,
            dates,
            [a.select(dates) for a in self.actions],
        )


class DateAction(Action):
    def __init__(self, context, action_path, **kwargs):
        super().__init__(context, action_path, **kwargs)

        datesconfig = {}
        subconfig = {}
        for k, v in deepcopy(kwargs).items():
            if k in ["start", "end", "frequency"]:
                datesconfig[k] = v
            else:
                subconfig[k] = v

        self._dates = build_groups(datesconfig)
        self.content = action_factory(subconfig, context)

    @trace_select
    def select(self, dates):
        newdates = self._dates.intersect(dates)
        if newdates.empty():
            return EmptyResult(self.context, None, newdates)
        return self.content.select(newdates)

    def __repr__(self):
        return super().__repr__(f"{self._dates}\n{self.content}")


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


def action_factory(config, context, action_path):
    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    # if len(config) == 2 and "label" in config:
    #     config = deepcopy(config)
    #     label = config.pop("label")
    #     return action_factory(
    #         dict(
    #             label=dict(
    #                 name=label,
    #                 **config,
    #             )
    #         ),
    #         context,
    #     )

    if len(config) != 1:
        raise ValueError(
            f"Invalid input config. Expecting dict with only one key, got {list(config.keys())}"
        )

    config = deepcopy(config)
    key = list(config.keys())[0]
    cls = dict(
        concat=ConcatAction,
        join=JoinAction,
        # label=LabelAction,
        pipe=PipeAction,
        # source=SourceAction,
        function=FunctionAction,
        dates=DateAction,
        # dependency=DependencyAction,
    ).get(key)

    if isinstance(config[key], list):
        args, kwargs = config[key], {}

    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    if cls is None:
        if not is_function(key, "actions"):
            raise ValueError(f"Unknown action {key}")
        cls = FunctionAction
        args = [key] + args

    return cls(context, action_path + [key], *args, **kwargs)


def step_factory(config, context, action_path, previous_step):
    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    config = deepcopy(config)
    assert len(config) == 1, config

    key = list(config.keys())[0]
    cls = dict(
        filter=FilterStepAction,
        # rename=RenameAction,
        # remapping=RemappingAction,
    ).get(key)

    if isinstance(config[key], list):
        args, kwargs = config[key], {}

    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    if cls is None:
        if not is_function(key, "steps"):
            raise ValueError(f"Unknown step {key}")
        cls = FunctionStepAction
        args = [key] + args

    return cls(context, action_path, previous_step, *args, **kwargs)


class FunctionContext:
    def __init__(self, owner):
        self.owner = owner

    def trace(self, emoji, *args):
        trace(emoji, *args)


class Context:
    def __init__(self, /, order_by, flatten_grid, remapping):
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)

        self.references = {}
        self.used_references = set()
        self.results = {}

    def register_reference(self, action_path, obj):
        assert isinstance(action_path, (list, tuple)), action_path
        action_path = tuple(action_path)
        trace("📚", step(action_path), "register", type(obj))
        if action_path in self.references:
            raise ValueError(f"Duplicate reference {action_path}")
        self.references[action_path] = obj

    def find_reference(self, action_path):
        assert isinstance(action_path, (list, tuple)), action_path
        action_path = tuple(action_path)
        if action_path in self.references:
            return self.references[action_path]
        # It can happend that the required action_path is not yet registered,
        # even if it is defined in the config.
        # Handling this case implies implementing a lazy inheritance resolution
        # and would complexify the code. This is not implemented.

        raise ValueError(f"Cannot find reference {action_path}")

    def will_need_reference(self, action_path):
        assert isinstance(action_path, (list, tuple)), action_path
        action_path = tuple(action_path)
        self.used_references.add(action_path)

    def notify_result(self, action_path, result):
        trace("🎯", step(action_path), "notify result", result)
        assert isinstance(action_path, (list, tuple)), action_path
        action_path = tuple(action_path)
        if action_path in self.used_references:
            if action_path in self.results:
                raise ValueError(f"Duplicate result {action_path}")
            self.results[action_path] = result

    def get_result(self, action_path):
        assert isinstance(action_path, (list, tuple)), action_path
        action_path = tuple(action_path)
        if action_path in self.results:
            return self.results[action_path]
        raise ValueError(f"Cannot find result {action_path}")


class InputBuilder:
    def __init__(self, config, **kwargs):
        self.kwargs = kwargs
        self.config = config
        self.action_path = ["input"]

    @trace_select
    def select(self, dates):
        """This changes the context."""
        dates = build_groups(dates)
        context = Context(**self.kwargs)
        action = action_factory(self.config, context, self.action_path)
        return action.select(dates)

    def __repr__(self):
        context = Context(**self.kwargs)
        a = action_factory(self.config, context, self.action_path)
        return repr(a)

    def _trace_select(self, dates):
        return f"InputBuilder({shorten(dates)})"


build_input = InputBuilder
