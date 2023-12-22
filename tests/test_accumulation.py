# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import os

import pytest

from ecml_tools.data import Subset, open_dataset

HAS_CREDENTIALS = os.path.exists(os.path.expanduser("~/.aws/credentials"))

ZARR = "s3://ml-datasets/stable/aifs-ea-an-oper-0001-mars-n320-1979-2022-1h-v4.zarr"


@pytest.mark.skipif(not HAS_CREDENTIALS, "No S3 credentials found")
def test_accumulation_1():
    ds = open_dataset(dataset=ZARR, accumulate=["tp", "cp"], frequency="6h")
    assert isinstance(ds, Subset), ds
    print(ds[0])


if __name__ == "__main__":
    test_accumulation_1()
