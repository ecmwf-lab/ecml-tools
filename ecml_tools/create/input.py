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
import os
import time
from collections import defaultdict
from copy import deepcopy
from functools import cached_property

from climetlab.core.order import build_remapping

from .group import build_groups
from .template import substitute
from .utils import seconds

LOG = logging.getLogger(__name__)


def find_function_path(name, kind):
    name = name.replace("-", "_")
    here = os.path.dirname(__file__)
    return os.path.join(here, "functions", kind, f"{name}.py")


def import_function(name, kind):
    path = find_function_path(name, kind)
    spec = importlib.util.spec_from_file_location(name, path)
    module = spec.loader.load_module()
    return module.execute


def is_function(name, kind):
    path = find_function_path(name, kind)
    return os.path.exists(path)


def assert_is_fieldset(obj):
    from climetlab.indexing.fieldset import FieldSet

    assert isinstance(obj, FieldSet), type(obj)


def _datasource_request(data):
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

    return dict(
        param_level=params_levels, param_step=params_steps, area=area, grid=grid
    )


class Cache:
    pass


class Coords:
    def __init__(self, owner):
        self.owner = owner
        self.cache = Cache()

    def _build_coords(self):
        # assert isinstance(self.owner.context, Context), type(self.owner.context)
        # assert isinstance(self.owner, Result), type(self.owner)
        # assert hasattr(self.owner, "context"), self.owner
        # assert hasattr(self.owner, "datasource"), self.owner
        # assert hasattr(self.owner, "get_cube"), self.owner
        # self.owner.datasource

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

        self.cache.variables = from_data[variables_key]  # "param_level"
        self.cache.ensembles = from_data[ensembles_key]  # "number"

        first_field = self.owner.datasource[0]
        grid_points = first_field.grid_points()
        grid_values = list(range(len(grid_points[0])))

        self.cache.grid_points = grid_points
        self.cache.resolution = first_field.resolution
        self.cache.grid_values = grid_values

    def __getattr__(self, name):
        if name in [
            "variables",
            "ensembles",
            "resolution",
            "grid_values",
            "grid_points",
        ]:
            if not hasattr(self.cache, name):
                self._build_coords()
            return getattr(self.cache, name)
        raise AttributeError(name)


class HasCoordsMixin:
    @property
    def variables(self):
        return self._coords.variables

    @property
    def ensembles(self):
        return self._coords.ensembles

    @property
    def resolution(self):
        return self._coords.resolution

    @property
    def grid_values(self):
        return self._coords.grid_values

    @property
    def grid_points(self):
        return self._coords.grid_points

    @property
    def dates(self):
        if self._dates is None:
            raise ValueError(f"No dates for {self}")
        return self._dates.values

    @property
    def frequency(self):
        return self._dates.frequency

    @property
    def shape(self):
        return [
            len(self.dates),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        ]

    @property
    def coords(self):
        return {
            "dates": self.dates,
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }


class Action:
    def __init__(self, context, /, *args, **kwargs):
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


class Result(HasCoordsMixin):
    empty = False

    def __init__(self, context, dates=None):
        assert isinstance(context, Context), type(context)
        self.context = context
        self._coords = Coords(self)
        self._dates = dates

    @property
    def datasource(self):
        self._raise_not_implemented()

    @property
    def data_request(self):
        """Returns a dictionary with the parameters needed to retrieve the data."""
        return _datasource_request(self.datasource)

    def get_cube(self):
        print(f"getting cube from {self.__class__.__name__}")
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


class EmptyResult(Result):
    empty = True

    def __init__(self, context, dates=None):
        super().__init__(context)

    @cached_property
    def datasource(self):
        from climetlab import load_source

        return load_source("empty")

    @property
    def variables(self):
        return []


class ReferencesSolver(dict):
    def __init__(self, context, dates):
        self.context = context
        self.dates = dates

    def __getitem__(self, key):
        if key == "dates":
            return self.dates.values
        if key in self.context.references:
            result = self.context.references[key]
            return result.datasource
        raise KeyError(key)


class FunctionResult(Result):
    def __init__(self, context, dates, action, previous_sibling=None):
        super().__init__(context, dates)
        assert isinstance(action, Action), type(action)
        self.action = action

        _args = self.action.args
        _kwargs = self.action.kwargs

        vars = ReferencesSolver(context, dates)

        self.args = substitute(_args, vars)
        self.kwargs = substitute(_kwargs, vars)

    # @cached_property
    @property
    def datasource(self):
        print(
            f"applying function {self.action.function} to {self.dates}, {self.args} {self.kwargs}, {self}"
        )
        return self.action.function(
            FunctionContext(self), self.dates, *self.args, **self.kwargs
        )

    def __repr__(self):
        content = " ".join([f"{v}" for v in self.args])
        content += " ".join([f"{k}={v}" for k, v in self.kwargs.items()])

        return super().__repr__(content)

    @property
    def function(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class JoinResult(Result):
    def __init__(self, context, dates, results, **kwargs):
        super().__init__(context, dates)
        self.results = [r for r in results if not r.empty]

    @property
    def datasource(self):
        ds = EmptyResult(self.context, self._dates).datasource
        for i in self.results:
            ds += i.datasource
            assert_is_fieldset(ds), i
        return ds

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class DependencyAction(Action):
    def __init__(self, context, **kwargs):
        super().__init__(context)
        self.content = action_factory(kwargs, context)

    def select(self, dates):
        self.content.select(dates)
        # this should trigger a registration of the result in the context
        # if there is a label
        # self.context.register_reference(self.name, result)
        return EmptyResult(self.context, dates)

    def __repr__(self):
        return super().__repr__(self.content)


class LabelAction(Action):
    def __init__(self, context, name, **kwargs):
        super().__init__(context)
        if len(kwargs) != 1:
            raise ValueError(f"Invalid kwargs for label : {kwargs}")
        self.name = name
        self.content = action_factory(kwargs, context)

    def select(self, dates):
        result = self.content.select(dates)
        self.context.register_reference(self.name, result)
        return result

    def __repr__(self):
        return super().__repr__(_inline_=self.name, _indent_=" ")


class FunctionAction(Action):
    def __init__(self, context, _name, **kwargs):
        super().__init__(context, **kwargs)
        self.name = _name

    def select(self, dates):
        return FunctionResult(self.context, dates, action=self)

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


class ConcatResult(Result):
    def __init__(self, context, results):
        super().__init__(context, dates=None)
        self.results = [r for r in results if not r.empty]

    @property
    def datasource(self):
        ds = EmptyResult(self.context, self.dates).datasource
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

    def __init__(self, context, *configs):
        super().__init__(context, *configs)
        self.actions = [action_factory(c, context) for c in configs]

    def __repr__(self):
        content = "\n".join([str(i) for i in self.actions])
        return super().__repr__(content)


class PipeAction(Action):
    def __init__(self, context, *configs):
        super().__init__(context, *configs)
        current = action_factory(configs[0], context)
        for c in configs[1:]:
            current = step_factory(c, context, _upstream_action=current)
        self.content = current

    def select(self, dates):
        return self.content.select(dates)

    def __repr__(self):
        return super().__repr__(self.content)


class StepResult(Result):
    def __init__(self, upstream, context, dates, action):
        super().__init__(context, dates)
        assert isinstance(upstream, Result), type(upstream)
        self.content = upstream
        self.action = action

    @property
    def datasource(self):
        return self.content.datasource


class StepAction(Action):
    result_class = None

    def __init__(self, context, _upstream_action, **kwargs):
        super().__init__(context, **kwargs)
        self.content = _upstream_action

    def select(self, dates):
        return self.result_class(
            self.content.select(dates),
            self.context,
            dates,
            self,
        )

    def __repr__(self):
        return super().__repr__(self.content, _inline_=str(self.kwargs))


class StepFunctionResult(StepAction):
    @property
    def datasource(self):
        return self.function(
            FunctionContext(self), self.content.datasource, **self.kwargs
        )


class FilterStepResult(StepResult):
    @property
    def datasource(self):
        ds = self.content.datasource
        assert_is_fieldset(ds)
        ds = ds.sel(**self.action.kwargs)
        assert_is_fieldset(ds)
        return ds


class FilterStepAction(StepAction):
    result_class = FilterStepResult


class ConcatAction(ActionWithList):
    def select(self, dates):
        return ConcatResult(self.context, [a.select(dates) for a in self.actions])


class JoinAction(ActionWithList):
    def select(self, dates):
        return JoinResult(self.context, dates, [a.select(dates) for a in self.actions])


class DateAction(Action):
    def __init__(self, context, **kwargs):
        super().__init__(context, **kwargs)

        datesconfig = {}
        subconfig = {}
        for k, v in deepcopy(kwargs).items():
            if k in ["start", "end", "frequency"]:
                datesconfig[k] = v
            else:
                subconfig[k] = v

        self._dates = build_groups(datesconfig)
        self.content = action_factory(subconfig, context)

    def select(self, dates):
        newdates = self._dates.intersect(dates)
        if newdates.empty():
            return EmptyResult(self.context, dates=newdates)
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


def action_factory(config, context):
    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    if len(config) == 2 and "label" in config:
        config = deepcopy(config)
        label = config.pop("label")
        return action_factory(
            dict(
                label=dict(
                    name=label,
                    **config,
                )
            ),
            context,
        )

    if len(config) != 1:
        raise ValueError(
            f"Invalid input config. Expecting dict with only one key, got {list(config.keys())}"
        )

    config = deepcopy(config)
    key = list(config.keys())[0]
    cls = dict(
        concat=ConcatAction,
        join=JoinAction,
        label=LabelAction,
        pipe=PipeAction,
        # source=SourceAction,
        function=FunctionAction,
        dates=DateAction,
        dependency=DependencyAction,
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

    return cls(context, *args, **kwargs)


def step_factory(config, context, _upstream_action):
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
    )[key]

    if isinstance(config[key], list):
        args, kwargs = config[key], {}

    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    if "_upstream_action" in kwargs:
        raise ValueError(f"Reserverd keyword '_upsream_action' in {config}")
    kwargs["_upstream_action"] = _upstream_action

    return cls(context, *args, **kwargs)


class FunctionContext:
    def __init__(self, owner):
        self.owner = owner


class Context:
    def __init__(self, /, order_by, flatten_grid, remapping):
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)

        self.references = {}

    def register_reference(self, name, obj):
        assert isinstance(obj, Result), type(obj)
        if name in self.references:
            raise ValueError(f"Duplicate reference {name}")
        self.references[name] = obj

    def find_reference(self, name):
        if name in self.references:
            return self.references[name]
        # It can happend that the required name is not yet registered,
        # even if it is defined in the config.
        # Handling this case implies implementing a lazy inheritance resolution
        # and would complexify the code. This is not implemented.
        raise ValueError(f"Cannot find reference {name}")


class InputBuilder:
    def __init__(self, config, **kwargs):
        self.kwargs = kwargs
        self.config = config

    def select(self, dates):
        """This changes the context."""
        dates = build_groups(dates)
        context = Context(**self.kwargs)
        action = action_factory(self.config, context)
        return action.select(dates)

    def __repr__(self):
        context = Context(**self.kwargs)
        a = action_factory(self.config, context)
        return repr(a)


build_input = InputBuilder
