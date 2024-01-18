# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import tempfile

from ecml_tools.create import Creator
from ecml_tools.data import open_dataset


def _config_path(name):
    here = os.path.dirname(__file__)
    return os.path.join(here, name + '.yaml')


def _output_path(name):
    here = os.path.dirname(__file__)
    return os.path.join(here, name + '-output')


def _reference_path(name):
    here = os.path.dirname(__file__)
    return os.path.join(here, name + '-reference')


def compare(dir1, dir2, skip=['.zattrs', 'provenance_load.json']):
    """Compare two directories recursively."""
    for file1 in os.listdir(dir1):
        path1 = os.path.join(dir1, file1)
        path2 = os.path.join(dir2, file1)
        if os.path.isfile(path1):
            assert os.path.isfile(path2), f"File {path2} does not exist"
            if file1 in skip:
                continue
            with open(path1, "rb") as f1:
                with open(path2, "rb") as f2:
                    assert f1.read() == f2.read(), f"{path1} != {path2}"
        else:
            assert os.path.isdir(path2), f"Directory {path2} does not exist"
            compare(path1, path2)


def test_create_1():
    output = _output_path("create-1")
    reference = _reference_path("create-1")
    config = _config_path("create-1")

    zarr_path = os.path.join(output, "create-1.zarr")

    c = Creator(zarr_path, config=config, cache=None, overwrite=True)
    # statistics_tmp=None,
    c.create()

    compare(output, reference)

    #ds = open_dataset(zarr_path)
    #assert ds.shape == (10, 8, 1, 162)
    #assert ds.variables == (10, 8, 1, 162)


if __name__ == "__main__":
    import sys

    test_create_1(sys.argv[1])
