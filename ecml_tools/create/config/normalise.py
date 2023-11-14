# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import warnings
from climetlab.core.order import build_remapping, normalize_order_by

class Purpose:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)

    def __call__(self, config):
        pass

    @classmethod
    def dict_to_str(cls, x):
        if isinstance(x, str):
            return x
        return list(x.keys())[0]


class NonePurpose(Purpose):
    def __call__(self, config):
        config.output.flatten_grid = config.output.get("flatten_grid", False)
        config.output.ensemble_dimension = config.output.get(
            "ensemble_dimension", False
        )


class AifsPurpose(Purpose):
    def __call__(self, config):
        def check_dict_value_and_set(dic, key, value):
            if key in dic:
                if dic[key] != value:
                    raise ValueError(
                        f"Cannot use {key}={dic[key]} with {self} purpose. Must use {value}."
                    )
            dic[key] = value

        def ensure_element_in_list(lst, elt, index):
            if elt in lst:
                assert lst[index] == elt
                return lst

            _lst = [self.dict_to_str(d) for d in lst]
            if elt in _lst:
                assert _lst[index] == elt
                return lst

            return lst[:index] + [elt] + lst[index:]

        check_dict_value_and_set(config.output, "flatten_grid", True)
        check_dict_value_and_set(config.output, "ensemble_dimension", 2)

        assert isinstance(config.output.order_by, (list, tuple)), config.output.order_by
        config.output.order_by = ensure_element_in_list(
            config.output.order_by, "number", config.output.ensemble_dimension
        )

        order_by = config.output.order_by
        assert len(order_by) == 3, order_by
        assert self.dict_to_str(order_by[0]) == "valid_datetime", order_by
        assert self.dict_to_str(order_by[2]) == "number", order_by


PURPOSES = {None: NonePurpose, "aifs": AifsPurpose}

def normalize_config(c):
    if "description" not in c:
        raise ValueError("Must provide a description in the config.")

    # deprecated/obsolete
    if "order" in c.output:
        raise ValueError(f"Do not use 'order'. Use order_by in {c}")
    if "loops" in c:
        warnings.warn("Should use loop instead of loops in config")
        assert "loop" not in c
        c.loop = c.pop("loops")
    
    func = PURPOSES[c.get("purpose")](c.get("purpose"))
    func(c)

    if not isinstance(c.input, (tuple, list)):
        LOG.warning(f"{c.input=} is not a list")
        c.input = [c.input]

    if not isinstance(c.loop, list):
        assert isinstance(c.loop, dict), c.loop
        c.loop = [dict(loop_a=c.loop)]

    if "order_by" in c.output:
        c.output.order_by = normalize_order_by(c.output.order_by)

    c.output.remapping = c.output.get("remapping", {})
    c.output.remapping = build_remapping(
        c.output.remapping, patches={"number": {None: 0}}
    )

    c.output.chunking = c.output.get("chunking", {})
    c.output.dtype = c.output.get("dtype", "float32")

    c.reading_chunks = c.get("reading_chunks")
    assert "flatten_values" not in c.output
    assert "flatten_grid" in c.output

    # The axis along which we append new data
    # TODO: assume grid points can be 2d as well
    c.output.append_axis = 0

    assert "statistics" in c.output
    statistics_axis_name = c.output.statistics
    statistics_axis = -1
    for i, k in enumerate(c.output.order_by):
        if k == statistics_axis_name:
            statistics_axis = i

    assert (
        statistics_axis >= 0
    ), f"{c.output.statistics} not in {list(c.output.order_by.keys())}"

    c.statistics_names = c.output.order_by[statistics_axis_name]

    # TODO: consider 2D grid points
    c.statistics_axis = statistics_axis