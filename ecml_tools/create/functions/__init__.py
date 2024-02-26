# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from collections import defaultdict


def assert_is_fieldset(obj):
    from climetlab.readers.grib.index import FieldSet

    assert isinstance(obj, FieldSet), type(obj)


def wrapped_mars_source(name, param, **kwargs):
    from climetlab import load_source

    assert name == "mars", name  # untested with other sources

    for_accumlated = dict(
        ea="era5-accumulations",
        oper="oper-accumulations",
        ei="oper-accumulations",
    )[kwargs["class"]]

    param_to_source = defaultdict(lambda: "mars")
    param_to_source.update(
        dict(
            tp=for_accumlated,
            cp=for_accumlated,
            lsp=for_accumlated,
        )
    )

    source_names = defaultdict(list)
    for p in param:
        source_names[param_to_source[p]].append(p)

    sources = []
    for n, params in source_names.items():
        sources.append(load_source(n, param=params, **patch_time_to_hours(kwargs)))
    return load_source("multi", sources)


def patch_time_to_hours(dic):
    # era5-accumulations requires time in hours
    if "time" not in dic:
        return dic
    time = dic["time"]
    assert isinstance(time, (tuple, list)), time
    time = [f"{int(t[:2]):02d}" for t in time]
    return {**dic, "time": time}
