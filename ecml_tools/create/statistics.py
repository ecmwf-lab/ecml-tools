# (C) Copyright 2023 ECMWF.
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
from collections import defaultdict

import numpy as np
from prepml.utils.text import table

from ecml_tools.provenance import gather_provenance_info

from .check import StatisticsValueError, check_data_values, check_stats

LOG = logging.getLogger(__name__)


class Registry:
    # names = [ "mean", "stdev", "minimum", "maximum", "sums", "squares", "count", ]
    # build_names = [ "minimum", "maximum", "sums", "squares", "count", ]
    version = 2

    def __init__(self, dirname, history_callback=None, overwrite=False):
        if history_callback is None:

            def dummy(*args, **kwargs):
                pass

            history_callback = dummy

        self.dirname = dirname
        self.overwrite = overwrite
        self.history_callback = history_callback

    def create(self, exist_ok):
        os.makedirs(self.dirname, exist_ok=exist_ok)

    def add_provenance(self, name="provenance", **kwargs):
        out = dict(provenance=gather_provenance_info(), **kwargs)
        with open(os.path.join(self.dirname, f"{name}.json"), "w") as f:
            json.dump(out, f)

    def delete(self):
        try:
            shutil.rmtree(self.dirname)
        except FileNotFoundError:
            pass

    def __setitem__(self, key, data):
        # if isinstance(key, slice):
        #     # this is just to make the filenames nicer.
        #     key_str = f"{key.start}_{key.stop}"
        #     if key.step is not None:
        #         key_str = f"{key_str}_{key.step}"
        # else:
        #     key_str = str(key_str)

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
            pickle.dump((key, data), f)
        shutil.move(tmp_path, path)

        LOG.info(f"Written {self.name} data for {key} in  {path}")

    def __iter__(self):
        # use glob to read all pickles
        files = glob.glob(self.dirname + "/*.npz")

        LOG.info(
            f"Reading {self.name} data, found {len(files)} for {self.name} in  {self.dirname}"
        )

        assert len(files) > 0, f"No files found in {self.dirname}"

        key_strs = dict()
        for f in files:
            with open(f, "rb") as f:
                key, data = pickle.load(f)

            key_str = str(key)
            if key_str in key_strs:
                raise Exception(
                    f"Duplicate key {key}, found in {f} and {key_strs[key_str]}"
                )
            key_strs[key_str] = f

            yield key, data

    def __str__(self):
        return f"Registry({self.dirname})"


class MissingDataException(Exception):
    pass


class StatisticsRegistry(Registry):
    name = "statistics"

    MissingDataException = MissingDataException

    def as_detailed_stats(self, shape):
        detailed_stats = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
        )

        flags = np.full(shape, False, dtype=np.bool_)
        for key, data in self:
            assert isinstance(data, dict), data
            assert not np.any(flags[key]), f"Overlapping values for {key} {flags}"
            flags[key] = True
            for name, array in detailed_stats.items():
                d = data[name]
                array[key] = d
        if not np.all(flags):
            missing_indexes = np.where(flags == False)  # noqa: E712
            raise self.MissingDataException(
                f"Missing statistics data for {missing_indexes}", missing_indexes
            )

        return detailed_stats


def compute_aggregated_statistics(data, variables_names):
    i_len = None
    for name, array in data.items():
        if i_len is None:
            i_len = len(array)
        assert len(array) == i_len, (name, len(array), i_len)

    for name, array in data.items():
        if name == "count":
            continue
        assert not np.isnan(array).any(), (name, array)

    # for i in range(0, i_len):
    #    for j in range(len(variables_names)):
    #        stats = Statistics(
    #            minimum=data["minimum"][i,:],
    #            maximum=data["maximum"][i,:],
    #            mean=data["mean"][i,:],
    #            stdev=data["stdev"][i,:],
    #            variables_names=variables_names,
    #        )

    _minimum = np.amin(data["minimum"], axis=0)
    _maximum = np.amax(data["maximum"], axis=0)
    _count = np.sum(data["count"], axis=0)
    _sums = np.sum(data["sums"], axis=0)
    _squares = np.sum(data["squares"], axis=0)
    _mean = _sums / _count

    assert all(_count[0] == c for c in _count), _count

    x = _squares / _count - _mean * _mean
    # remove negative variance due to numerical errors
    # x[- 1e-15 < (x / (np.sqrt(_squares / _count) + np.abs(_mean))) < 0] = 0
    check_variance_is_positive(
        x, variables_names, _minimum, _maximum, _mean, _count, _sums, _squares
    )
    _stdev = np.sqrt(x)

    stats = Statistics(
        minimum=_minimum,
        maximum=_maximum,
        mean=_mean,
        count=_count,
        sums=_sums,
        squares=_squares,
        stdev=_stdev,
        variables_names=variables_names,
    )

    return stats


def compute_statistics(array, check_variables_names=None):
    nvars = array.shape[1]
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
        stats = [self[name] for name in self.STATS_NAMES]

        rows = []

        for i, v in enumerate(self["variables_names"]):
            rows.append([i, v] + [x[i] for x in stats])

        return table(
            rows,
            header=["Index", "Variable", "Min", "Max", "Mean", "Stdev"],
            align=[">", "<", ">", ">", ">", ">"],
            margin=3,
        )

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
        with open(filename, "r") as f:
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


def check_variance_is_positive(
    x, variables_names, minimum, maximum, mean, count, sums, squares
):
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
