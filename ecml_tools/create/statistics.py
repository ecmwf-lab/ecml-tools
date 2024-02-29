# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import glob
import json
import logging
import os
import pickle
import shutil
import socket
from collections import Counter, defaultdict
from functools import cached_property

import numpy as np

from ecml_tools.provenance import gather_provenance_info

from .check import StatisticsValueError, check_data_values, check_stats

LOG = logging.getLogger(__name__)


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


def compute_statistics(array, check_variables_names=None):
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
        minimum[i] = np.min(values, axis=1)
        maximum[i] = np.max(values, axis=1)
        sums[i] = np.sum(values, axis=1)
        squares[i] = np.sum(values * values, axis=1)
        count[i] = values.shape[1]

        for j, name in enumerate(check_variables_names):
            check_data_values(values[j, :], name=name)

    return {
        "minimum": minimum,
        "maximum": maximum,
        "sums": sums,
        "squares": squares,
        "count": count,
    }


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
        key_str = (
            str(key)
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace(",", "_")
            .replace("None", "x")
            .replace("__", "_")
            .lower()
        )
        path = os.path.join(self.dirname, f"{key_str}.npz")

        if not self.overwrite:
            assert not os.path.exists(path), f"{path} already exists"

        tmp_path = path + f".tmp-{os.getpid()}-on-{socket.gethostname()}"
        with open(tmp_path, "wb") as f:
            pickle.dump((key, dates, data), f)
        shutil.move(tmp_path, path)

        LOG.info(f"Written statistics data for {key} in {path} ({dates})")

    def _gather_data(self):
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.npz")
        LOG.info(f"Reading stats data, found {len(files)} in {self.dirname}")
        assert len(files) > 0, f"No files found in {self.dirname}"

        key_strs = dict()
        for f in files:
            with open(f, "rb") as f:
                key, dates, data = pickle.load(f)

            key_str = str(key)
            if key_str in key_strs:
                raise Exception(
                    f"Duplicate key {key}, found in {f} and {key_strs[key_str]}"
                )
            key_strs[key_str] = f

            yield key, dates, data

    @cached_property
    def n_dates_computed(self):
        return len(self.dates_computed)

    @property
    def dates_computed(self):
        all_dates = []
        for key, dates, data in self._gather_data():
            all_dates += dates

        # assert no duplicates
        duplicates = [item for item, count in Counter(all_dates).items() if count > 1]
        if duplicates:
            raise StatisticsValueError(
                f"Duplicate dates found in statistics: {duplicates}"
            )

        all_dates = normalise_dates(all_dates)
        return all_dates

    def get_aggregated(self, dates, variables_names):
        aggregator = StatAggregator(variables_names, self)
        aggregator.read(dates)
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

    def __init__(self, variables_names, owner):
        self.owner = owner
        self.computed_dates = owner.dates_computed
        self.shape = (len(self.computed_dates), len(variables_names))
        self.variables_names = variables_names
        print("Aggregating on ", self.shape, variables_names)

        self.minimum = np.full(self.shape, np.nan, dtype=np.float64)
        self.maximum = np.full(self.shape, np.nan, dtype=np.float64)
        self.sums = np.full(self.shape, np.nan, dtype=np.float64)
        self.squares = np.full(self.shape, np.nan, dtype=np.float64)
        self.count = np.full(self.shape, -1, dtype=np.int64)
        self.flags = np.full(self.shape, False, dtype=np.bool_)

    def read(self, dates):
        assert type(dates[0]) is type(self.computed_dates[0]), (
            dates[0],
            self.computed_dates[0],
        )

        dates_bitmap = np.isin(self.computed_dates, dates)

        for key, dates, data in self.owner._gather_data():
            assert isinstance(data, dict), data
            assert not np.any(
                self.flags[key]
            ), f"Overlapping values for {key} {self.flags} ({dates})"
            self.flags[key] = True
            for name in self.NAMES:
                array = getattr(self, name)
                array[key] = data[name]

        if not np.all(self.flags[dates_bitmap]):
            not_found = np.where(self.flags == False)  # noqa: E712
            raise Exception(f"Missing statistics data for {not_found}", not_found)

        print(
            f"Selection statistics data from {self.minimum.shape[0]} to {self.minimum[dates_bitmap].shape[0]} dates."
        )
        for name in self.NAMES:
            array = getattr(self, name)
            array = array[dates_bitmap]
            setattr(self, name, array)

    def aggregate(self):
        print(f"Aggregating statistics on {self.minimum.shape}")
        for name in self.NAMES:
            if name == "count":
                continue
            array = getattr(self, name)
            assert not np.isnan(array).any(), (name, array)

        minimum = np.amin(self.minimum, axis=0)
        maximum = np.amax(self.maximum, axis=0)
        count = np.sum(self.count, axis=0)
        sums = np.sum(self.sums, axis=0)
        squares = np.sum(self.squares, axis=0)
        mean = sums / count

        assert all(count[0] == c for c in count), count

        x = squares / count - mean * mean
        # remove negative variance due to numerical errors
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
        check_variance(
            x, self.variables_names, minimum, maximum, mean, count, sums, squares
        )
        stdev = np.sqrt(x)

        stats = Statistics(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.variables_names,
        )

        return stats


class Statistics(dict):
    STATS_NAMES = ["minimum", "maximum", "mean", "stdev"]  # order matter for __str__.

    def __init__(self, *args, check=True, **kwargs):
        super().__init__(*args, **kwargs)
        if check:
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
        header = ["Variables"] + [self[name] for name in self.STATS_NAMES]
        out = " ".join(header)

        for i, v in enumerate(self["variables_names"]):
            out += " ".join([v] + [f"{x[i]:.2f}" for x in self.values()])
        return out

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
