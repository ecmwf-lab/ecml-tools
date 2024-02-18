#!/usr/bin/env python3
# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import json
import os

import numpy as np

from ecml_tools.create import Creator
from ecml_tools.data import open_dataset


def compare_dot_zattrs(a, b):
    if isinstance(a, dict):
        a_keys = list(a.keys())
        b_keys = list(b.keys())
        for k in set(a_keys) & set(b_keys):
            if k in ["timestamp", "uuid", "latest_write_timestamp", "yaml_config"]:
                assert type(a[k]) == type(b[k]), (  # noqa: E721
                    type(a[k]),
                    type(b[k]),
                    a[k],
                    b[k],
                )
            assert k in a_keys, (k, a_keys)
            assert k in b_keys, (k, b_keys)
            return compare_dot_zattrs(a[k], b[k])

    if isinstance(a, list):
        assert len(a) == len(b), (a, b)
        for v, w in zip(a, b):
            return compare_dot_zattrs(v, w)

    assert type(a) == type(b), (type(a), type(b), a, b)  # noqa: E721
    return a == b, (a, b)


def compare_zarr(dir1, dir2):
    a = open_dataset(dir1)
    b = open_dataset(dir2)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a.dates == b.dates).all(), (a.dates, b.dates)
    for a_, b_ in zip(a.variables, b.variables):
        assert a_ == b_, (a, b)
    for i_date, date in zip(range(a.shape[0]), a.dates):
        for i_param in range(a.shape[1]):
            param = a.variables[i_param]
            assert param == b.variables[i_param], (
                date,
                param,
                a.variables[i_param],
                b.variables[i_param],
            )
            a_ = a[i_date, i_param]
            b_ = b[i_date, i_param]
            assert a.shape == b.shape, (date, param, a.shape, b.shape)
            delta = a_ - b_
            max_delta = np.max(np.abs(delta))
            assert max_delta == 0.0, (date, param, a_, b_, a_ - b_, max_delta)
    compare(dir1, dir2)


def compare(dir1, dir2):
    """Compare two directories recursively."""
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    files = files1.union(files2)

    for f in files:
        path1 = os.path.join(dir1, f)
        path2 = os.path.join(dir2, f)

        if not os.path.isfile(path1):
            assert os.path.isdir(path2), f"Directory {path2} does not exist"
            compare(path1, path2)
            continue

        assert os.path.isfile(path2), f"File {path2} does not exist"

        content1 = open(path1, "rb").read()
        content2 = open(path2, "rb").read()

        if f == ".zattrs":
            compare_dot_zattrs(json.loads(content1), json.loads(content2))
            continue

        if f == "provenance_load.json":
            assert False, "test not implemented to compare temporary statistics"

        assert content1 == content2, f"{path1} != {path2}"


def _test_create(name):
    here = os.path.dirname(__file__)
    config = os.path.join(here, name + ".yaml")
    output = os.path.join(here, name + "-output", name + ".zarr")
    reference = os.path.join(here, name + "-reference", name + ".zarr")

    # cache=None is using the default cache
    c = Creator(
        output,
        config=config,
        cache=None,
        overwrite=True,
    )
    c.create()

    compare_zarr(reference, output)


def test_create_concat():
    _test_create("create-concat")


def test_create_join():
    _test_create("create-join")


def test_create_pipe():
    _test_create("create-pipe")


def test_create_perturbations():
    _test_create("create-perturbations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the test case")
    parser.add_argument("--compare", nargs=2, help="Compare two directories")

    args = parser.parse_args()

    if args.compare:
        compare_zarr(args.compare[0], args.compare[1])
    else:
        _test_create(args.name)
