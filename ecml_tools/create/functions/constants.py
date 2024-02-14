# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from copy import deepcopy

from climetlab import load_source

from ecml_tools.create.utils import to_datetime_list

DEBUG = True


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def get_template_field(request):
    """Create a template request from the initial request, setting the date, time,
    levtype and param fields."""
    template_request = {
        "class": "ea",
        "expver": "0001",
        "type": "an",
        "date": "20200101",
        "time": "0000",
        "levtype": "sfc",
        "param": "2t",
    }
    for k in ["area", "grid"]:  # is class needed?
        if k in request:
            template_request[k] = request[k]
    template = load_source("mars", template_request)
    assert len(template) == 1, (len(template), template_request)
    return template


def normalise_time_to_hours(r):
    r = deepcopy(r)
    if "time" not in r:
        return r

    times = []
    for t in to_list(r["time"]):
        assert len(t) == 4, r
        assert t.endswith("00"), r
        times.append(int(t) // 100)
    r["time"] = tuple(times)
    return r


def constants(context, dates, **request):
    template = get_template_field(request)

    print(f"✅ load_source(constants, {template}, {request}")
    return load_source("constants", source_or_dataset=template, **request)


execute = constants

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
      class: ea
      expver: '0001'
      grid: 20.0/20.0
      levtype: sfc
      # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]
      number: [0, 1]
      param: [cos_latitude]
    """
    )
    dates = yaml.safe_load(
        "[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]"
    )
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in constants(None, dates, **config):
        print(f, f.to_numpy().mean())
