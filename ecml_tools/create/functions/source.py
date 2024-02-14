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

from ecml_tools.create.input import Context
from ecml_tools.create.utils import to_datetime_list

DEBUG = True


def source(context, dates, **kwargs):
    name = kwargs.pop("name")
    print(f"✅ load_source({name}, {dates}, {kwargs}")
    date = list({d.strftime("%Y%m%d") for d in dates})
    time = list({d.strftime("%H%M") for d in dates})
    kwargs["date"] = kwargs.get("date", date)
    kwargs["time"] = kwargs.get("time", time)
    return load_source(name, **kwargs)


execute = source

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
    - class: ea
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
    for f in constants(None, dates, *config):
        print(f, f.to_numpy().mean())
