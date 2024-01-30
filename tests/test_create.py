# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import tempfile
import json

from ecml_tools.create import Creator
from ecml_tools.data import open_dataset


def compare_dot_zattrs(a, b):
    if isinstance(a, dict):
        for k in a:
            assert k in b
            if k in ["timestamp", "uuid", "latest_write_timestamp"]:
                assert type(a[k]) == type(b[k]), (type(a[k]), type(b[k]), a[k], b[k])
            return compare_dot_zattrs(a[k], b[k])

    if isinstance(a, list):
        assert len(a) == len(b), (a, b)
        for v, w in zip(a, b):
            return compare_dot_zattrs(v, w)

    assert type(a) == type(b), (type(a), type(b), a, b)
    return a == b, (a, b)


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

        content1 = open(path1,'rb').read()
        content2 = open(path2,'rb').read()

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

    c = Creator(output, config=config, cache=None, overwrite=True)
    c.create()

    # ds = open_dataset(zarr_path)
    # assert ds.shape == 
    # assert ds.variables == 

    compare(reference, output)



def test_create_concat():
    _test_create("create-concat")


def test_create_join():
    _test_create("create-join")


def test_create_pipe():
    _test_create("create-pipe")


if __name__ == "__main__":
    import sys

    _test_create(sys.argv[1])
