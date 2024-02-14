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

from ecml_tools.create.functions.mars import factorise_requests
from ecml_tools.create.utils import to_datetime_list

DEBUG = True


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


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


def accumulations(context, dates, **request):
    to_list(request["param"])
    class_ = request["class"]

    source_name = dict(
        ea="era5-accumulations",
        oper="oper-accumulations",
        ei="oper-accumulations",
    )[class_]

    requests = factorise_requests(dates, request)

    ds = load_source("empty")
    for r in requests:
        r = {k: v for k, v in r.items() if v != ("-",)}
        r = normalise_time_to_hours(r)

        if DEBUG:
            print(f"load_source({source_name},  {r}")
        ds = ds + load_source(source_name, **r)
    return ds


execute = accumulations

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
      class: ea
      expver: '0001'
      grid: 20./20.
      levtype: sfc
#      number: [0, 1]
#      stream: enda
      param: [cp, tp]
#      accumulation_period: 6h
    """
    )
    dates = yaml.safe_load(
        "[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]"
    )
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in accumulations(None, dates, **config):
        print(f, f.to_numpy().mean())
