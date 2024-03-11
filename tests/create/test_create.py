#!/usr/bin/env python3
# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import glob
import os

import numpy as np
import pytest
import zarr

from ecml_tools.create import Creator
from ecml_tools.data import open_dataset

HERE = os.path.dirname(__file__)
# find_yamls
NAMES = [os.path.basename(path).split(".")[0] for path in glob.glob(os.path.join(HERE, "*.yaml"))]
NAMES = [
    name
    for name in NAMES
    if name
    not in [
        "perturbations",
    ]
]
assert NAMES, "No yaml files found in " + HERE

TEST_DATA_ROOT = "s3://ml-tests/test-data/anemoi-datasets/create/"


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


def compare_datasets(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a.dates == b.dates).all(), (a.dates, b.dates)
    for a_, b_ in zip(a.variables, b.variables):
        assert a_ == b_, (a, b)
    assert a.missing == b.missing, "Missing are different"

    for i_date, date in zip(range(a.shape[0]), a.dates):
        if i_date in a.missing:
            continue
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

            a_nans = np.isnan(a_)
            b_nans = np.isnan(b_)
            assert np.all(a_nans == b_nans), (date, param, "nans are different")

            a_ = np.where(a_nans, 0, a_)
            b_ = np.where(b_nans, 0, b_)

            delta = a_ - b_
            max_delta = np.max(np.abs(delta))
            assert max_delta == 0.0, (date, param, a_, b_, a_ - b_, max_delta)


def compare_statistics(ds1, ds2):
    vars1 = ds1.variables
    vars2 = ds2.variables
    assert len(vars1) == len(vars2)
    for v1, v2 in zip(vars1, vars2):
        idx1 = ds1.name_to_index[v1]
        idx2 = ds2.name_to_index[v2]
        assert (ds1.statistics["mean"][idx1] == ds2.statistics["mean"][idx2]).all()
        assert (ds1.statistics["stdev"][idx1] == ds2.statistics["stdev"][idx2]).all()
        assert (ds1.statistics["maximum"][idx1] == ds2.statistics["maximum"][idx2]).all()
        assert (ds1.statistics["minimum"][idx1] == ds2.statistics["minimum"][idx2]).all()


class Comparer:
    def __init__(self, name, output_path=None, reference_path=None):
        self.name = name
        self.reference = reference_path or os.path.join(TEST_DATA_ROOT, name + ".zarr")
        self.output = output_path or os.path.join(name + ".zarr")
        print(f"Comparing {self.reference} and {self.output}")

        self.z_reference = zarr.open(self.reference)
        self.z_output = zarr.open(self.output)

        self.ds_reference = open_dataset(self.reference)
        self.ds_output = open_dataset(self.output)

    def compare(self):
        compare_dot_zattrs(self.z_output.attrs, self.z_reference.attrs)
        compare_datasets(self.ds_output, self.ds_reference)
        compare_statistics(self.ds_output, self.ds_reference)


@pytest.mark.parametrize("name", NAMES)
def test_run(name):
    config = os.path.join(HERE, name + ".yaml")
    output = os.path.join(HERE, name + ".zarr")
    comparer = Comparer(name, output_path=output)

    # cache=None is using the default cache
    c = Creator(
        output,
        config=config,
        cache=None,
        overwrite=True,
    )
    c.create()

    comparer.compare()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the test case")

    args = parser.parse_args()

    test_run(args.name)
