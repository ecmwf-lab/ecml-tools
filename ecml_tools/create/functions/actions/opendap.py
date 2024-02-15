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


def opendap(context, dates, url_pattern, *args, **kwargs):

    all_urls = Pattern(url_pattern, ignore_missing_keys=True).substitute(
        *args, date=dates, **kwargs
    )

    ds = load_source("empty")
    levels = kwargs.get("level", kwargs.get("levelist"))

    for url in all_urls:

        print("URL", url)
        s = load_source("opendap", url)
        s = s.sel(
            valid_datetime=[d.isoformat() for d in dates],
            param=kwargs["param"],
            step=kwargs.get("step", 0),
        )
        if levels:
            s = s.sel(levelist=levels)
        ds = ds + s
    return ds


execute = opendap
