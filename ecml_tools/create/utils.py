# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import os
import numpy as np


def compute_directory_size(path):
    if not os.path.isdir(path):
        return None
    size = 0
    n = 0
    for dirpath, _, filenames in tqdm.tqdm(
        os.walk(path), desc="Computing size", leave=False
    ):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            size += os.path.getsize(file_path)
            n += 1
    return size, n


def _prepare_serialisation(o):
    if isinstance(o, dict):
        dic = {}
        for k, v in o.items():
            v = _prepare_serialisation(v)
            if k == "order_by":
                # zarr attributes are saved with sort_keys=True
                # and ordered dict are reordered.
                # This is a problem for "order_by"
                # We ensure here that the order_by key contains
                # a list of dict
                v = [{kk: vv} for kk, vv in v.items()]
            dic[k] = v
        return dic

    if isinstance(o, (list, tuple)):
        return [_prepare_serialisation(v) for v in o]

    if o in (None, True, False):
        return o

    if isinstance(o, (str, int, float)):
        return o

    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

    return str(o)


def normalize_and_check_dates(dates, start, end, frequency, dtype="datetime64[s]"):
    assert isinstance(frequency, int), frequency
    start = np.datetime64(start)
    end = np.datetime64(end)
    delta = np.timedelta64(frequency, "h")

    res = []
    while start <= end:
        res.append(start)
        start += delta
    dates_ = np.array(res).astype(dtype)

    if len(dates_) != len(dates):
        raise ValueError(
            f"Final date size {len(dates_)} (from {dates_[0]} to {dates_[-1]}, "
            f"{frequency=}) does not match data shape {len(dates)} (from {dates[0]} to "
            f"{dates[-1]})."
        )

    for i, (d1, d2) in enumerate(zip(dates, dates_)):
        assert d1 == d2, (i, d1, d2)

    return dates_
