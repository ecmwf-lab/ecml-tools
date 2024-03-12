# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import glob
import hashlib
import json
import logging
import os
import pickle
import shutil
import socket
from collections import defaultdict

import numpy as np

from ecml_tools.provenance import gather_provenance_info

from .check import StatisticsValueError
from .check import check_data_values
from .check import check_stats

LOG = logging.getLogger(__name__)


def to_datetime(date):
    if isinstance(date, str):
        return np.datetime64(date)
    if isinstance(date, datetime.datetime):
        return np.datetime64(date)
    return date


def to_datetimes(dates):
    return [to_datetime(d) for d in dates]


def check_variance(x, variables_names, minimum, maximum, mean, count, sums, squares):
    if (x >= 0).all():
        return
    print(x)
    print(variables_names)
    print(count)
    for i, (var, y) in enumerate(zip(variables_names, x)):
        if y >= 0:
            continue
        print(
            var,
            y,
            maximum[i],
            minimum[i],
            mean[i],
            count[i],
            sums[i],
            squares[i],
        )

        print(var, np.min(sums[i]), np.max(sums[i]), np.argmin(sums[i]))
        print(var, np.min(squares[i]), np.max(squares[i]), np.argmin(squares[i]))
        print(var, np.min(count[i]), np.max(count[i]), np.argmin(count[i]))

    raise ValueError("Negative variance")


def compute_statistics(array, check_variables_names=None, allow_nan=False):
    nvars = array.shape[1]

    print("Stats", nvars, array.shape, check_variables_names)
    if check_variables_names:
        assert nvars == len(check_variables_names), (nvars, check_variables_names)
    stats_shape = (array.shape[0], nvars)

    count = np.zeros(stats_shape, dtype=np.int64)
    sums = np.zeros(stats_shape, dtype=np.float64)
    squares = np.zeros(stats_shape, dtype=np.float64)
    minimum = np.zeros(stats_shape, dtype=np.float64)
    maximum = np.zeros(stats_shape, dtype=np.float64)

    for i, chunk in enumerate(array):
        values = chunk.reshape((nvars, -1))

        for j, name in enumerate(check_variables_names):
            check_data_values(values[j, :], name=name, allow_nan=allow_nan)
            if np.isnan(values[j, :]).all():
                # LOG.warning(f"All NaN values for {name} ({j}) for date {i}")
                raise ValueError(f"All NaN values for {name} ({j}) for date {i}")

        # Ignore NaN values
        minimum[i] = np.nanmin(values, axis=1)
        maximum[i] = np.nanmax(values, axis=1)
        sums[i] = np.nansum(values, axis=1)
        squares[i] = np.nansum(values * values, axis=1)
        count[i] = np.sum(~np.isnan(values), axis=1)

    return {"minimum": minimum, "maximum": maximum, "sums": sums, "squares": squares, "count": count}


class TempStatistics:
    version = 3
    # Used in parrallel, during data loading,
    # to write statistics in pickled npz files.
    # can provide statistics for a subset of dates.

    def __init__(self, dirname, overwrite=False):
        self.dirname = dirname
        self.overwrite = overwrite

    def add_provenance(self, **kwargs):
        self.create(exist_ok=True)
        out = dict(provenance=gather_provenance_info(), **kwargs)
        with open(os.path.join(self.dirname, "provenance.json"), "w") as f:
            json.dump(out, f)

    def create(self, exist_ok):
        os.makedirs(self.dirname, exist_ok=exist_ok)

    def delete(self):
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def write(self, key, data, dates):
        self.create(exist_ok=True)
        h = hashlib.sha256(str(dates).encode("utf-8")).hexdigest()
        path = os.path.join(self.dirname, f"{h}.npz")

        if not self.overwrite:
            assert not os.path.exists(path), f"{path} already exists"

        tmp_path = path + f".tmp-{os.getpid()}-on-{socket.gethostname()}"
        with open(tmp_path, "wb") as f:
            pickle.dump((key, dates, data), f)
        shutil.move(tmp_path, path)

        LOG.info(f"Written statistics data for {len(dates)} dates in {path} ({dates})")

    def _gather_data(self):
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.npz")
        LOG.info(f"Reading stats data, found {len(files)} files in {self.dirname}")
        assert len(files) > 0, f"No files found in {self.dirname}"
        for f in files:
            with open(f, "rb") as f:
                yield pickle.load(f)

    def get_aggregated(self, *args, **kwargs):
        aggregator = StatAggregator(self, *args, **kwargs)
        return aggregator.aggregate()

    def __str__(self):
        return f"TempStatistics({self.dirname})"


def normalise_date(d):
    if isinstance(d, str):
        d = np.datetime64(d)
    return d


def normalise_dates(dates):
    return [normalise_date(d) for d in dates]


class StatAggregator:
    NAMES = ["minimum", "maximum", "sums", "squares", "count"]

    def __init__(self, owner, dates, variables_names, allow_nan):
        dates = sorted(dates)
        dates = to_datetimes(dates)
        self.owner = owner
        self.dates = dates
        self.variables_names = variables_names
        self.allow_nan = allow_nan

        self.shape = (len(self.dates), len(self.variables_names))
        print("Aggregating statistics on ", self.shape, self.variables_names)

        self.minimum = np.full(self.shape, np.nan, dtype=np.float64)
        self.maximum = np.full(self.shape, np.nan, dtype=np.float64)
        self.sums = np.full(self.shape, np.nan, dtype=np.float64)
        self.squares = np.full(self.shape, np.nan, dtype=np.float64)
        self.count = np.full(self.shape, -1, dtype=np.int64)

        self._read()

    def _read(self):
        def check_type(a, b):
            a = list(a)
            b = list(b)
            a = a[0] if a else None
            b = b[0] if b else None
            assert type(a) is type(b), (type(a), type(b))

        found = set()
        offset = 0
        for _, _dates, stats in self.owner._gather_data():
            assert isinstance(stats, dict), stats
            assert stats["minimum"].shape[0] == len(_dates), (stats["minimum"].shape, len(_dates))
            assert stats["minimum"].shape[1] == len(self.variables_names), (
                stats["minimum"].shape,
                len(self.variables_names),
            )
            for n in self.NAMES:
                assert n in stats, (n, list(stats.keys()))
            _dates = to_datetimes(_dates)
            check_type(_dates, self.dates)
            if found:
                check_type(found, self.dates)
                assert found.isdisjoint(_dates), "Duplicate dates found in precomputed statistics"

            # filter dates
            dates = set(_dates) & set(self.dates)

            if not dates:
                # dates have been completely filtered for this chunk
                continue

            # filter data
            bitmap = np.isin(_dates, self.dates)
            for k in self.NAMES:
                stats[k] = stats[k][bitmap]

            assert stats["minimum"].shape[0] == len(dates), (stats["minimum"].shape, len(dates))

            # store data in self
            found |= set(dates)
            for name in self.NAMES:
                array = getattr(self, name)
                assert stats[name].shape[0] == len(dates), (stats[name].shape, len(dates))
                array[offset : offset + len(dates)] = stats[name]
            offset += len(dates)

        for d in self.dates:
            assert d in found, f"Statistics for date {d} not precomputed."
        assert len(self.dates) == len(found), "Not all dates found in precomputed statistics"
        assert len(self.dates) == offset, "Not all dates found in precomputed statistics."
        print(f"Statistics for {len(found)} dates found.")

    def aggregate(self):

        minimum = np.nanmin(self.minimum, axis=0)
        maximum = np.nanmax(self.maximum, axis=0)
        sums = np.nansum(self.sums, axis=0)
        squares = np.nansum(self.squares, axis=0)
        count = np.nansum(self.count, axis=0)
        mean = sums / count

        assert sums.shape == count.shape == squares.shape == mean.shape == minimum.shape == maximum.shape

        x = squares / count - mean * mean
        # remove negative variance due to numerical errors
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
        check_variance(x, self.variables_names, minimum, maximum, mean, count, sums, squares)
        stdev = np.sqrt(x)

        for j, name in enumerate(self.variables_names):
            check_data_values(
                np.array(
                    [
                        mean[j],
                    ]
                ),
                name=name,
                allow_nan=False,
            )

        return Statistics(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.variables_names,
        )


class Statistics(dict):
    STATS_NAMES = ["minimum", "maximum", "mean", "stdev"]  # order matter for __str__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check()

    @property
    def size(self):
        return len(self["variables_names"])

    def check(self):
        for k, v in self.items():
            if k == "variables_names":
                assert len(v) == self.size
                continue
            assert v.shape == (self.size,)
            if k == "count":
                assert (v >= 0).all(), (k, v)
                assert v.dtype == np.int64, (k, v)
                continue
            if k == "stdev":
                assert (v >= 0).all(), (k, v)
            assert v.dtype == np.float64, (k, v)

        for i, name in enumerate(self["variables_names"]):
            try:
                check_stats(**{k: v[i] for k, v in self.items()}, msg=f"{i} {name}")
                check_data_values(self["minimum"][i], name=name)
                check_data_values(self["maximum"][i], name=name)
                check_data_values(self["mean"][i], name=name)
            except StatisticsValueError as e:
                e.args += (i, name)
                raise

    def __str__(self):
        header = ["Variables"] + self.STATS_NAMES
        out = [" ".join(header)]

        out += [
            " ".join([v] + [f"{self[n][i]:.2f}" for n in self.STATS_NAMES])
            for i, v in enumerate(self["variables_names"])
        ]
        return "\n".join(out)

    def save(self, filename, provenance=None):
        assert filename.endswith(".json"), filename
        dic = {}
        for k in self.STATS_NAMES:
            dic[k] = list(self[k])

        out = dict(data=defaultdict(dict))
        for i, name in enumerate(self["variables_names"]):
            for k in self.STATS_NAMES:
                out["data"][name][k] = dic[k][i]

        out["provenance"] = provenance

        with open(filename, "w") as f:
            json.dump(out, f, indent=2)

    def load(self, filename):
        assert filename.endswith(".json"), filename
        with open(filename) as f:
            dic = json.load(f)

        dic_ = {}
        for k, v in dic.items():
            if k == "count":
                dic_[k] = np.array(v, dtype=np.int64)
                continue
            if k == "variables":
                dic_[k] = v
                continue
            dic_[k] = np.array(v, dtype=np.float64)
        return Statistics(dic_)
