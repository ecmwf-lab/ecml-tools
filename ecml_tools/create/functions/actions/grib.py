# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab import load_source
from climetlab.utils.patterns import Pattern


def execute(context, dates, path, *args, **kwargs):
    paths = Pattern(path, ignore_missing_keys=True).substitute(
        *args, date=dates, **kwargs
    )

    ds = load_source("empty")

    for path in paths:
        print("PATH", path)
        s = load_source("file", path)
        s = s.sel(valid_datetime=[d.isoformat() for d in dates], **kwargs)

        ds = ds + s
    return ds
