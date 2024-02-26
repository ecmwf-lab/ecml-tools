# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import warnings
from copy import deepcopy

import numpy as np
import tqdm
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output

from ecml_tools.create.check import check_data_values
from ecml_tools.create.functions import assert_is_fieldset, wrapped_mars_source


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    if isinstance(x, str):
        return x.split("/")
    return [x]


def normalise_number(number):
    number = to_list(number)

    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    return number


def normalise_request(request):
    request = deepcopy(request)
    if "number" in request:
        request["number"] = normalise_number(request["number"])
    if "time" in request:
        request["time"] = to_list(request["time"])
    request["param"] = to_list(request["param"])
    return request


def ensembles_perturbations(ensembles, center, mean, remapping={}, patches={}):
    from climetlab import load_source

    ensembles = normalise_request(ensembles)
    center = normalise_request(center)
    mean = normalise_request(mean)

    number_list = ensembles["number"]
    n_numbers = len(number_list)

    keys = ["param", "level", "valid_datetime", "date", "time", "step", "number"]

    print(f"Retrieving ensemble data with {ensembles}")
    ensembles = wrapped_mars_source(**ensembles).order_by(*keys)
    print(f"Retrieving center data with {center}")
    center = wrapped_mars_source(**center).order_by(*keys)
    print(f"Retrieving mean data with {mean}")
    mean = wrapped_mars_source(**mean).order_by(*keys)

    assert len(mean) * n_numbers == len(ensembles), (
        len(mean),
        n_numbers,
        len(ensembles),
    )
    assert len(center) * n_numbers == len(ensembles), (
        len(center),
        n_numbers,
        len(ensembles),
    )

    # prepare output tmp file so we can read it back
    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    for i, field in tqdm.tqdm(enumerate(ensembles)):
        param = field.metadata("param")
        number = field.metadata("number")
        ii = i // n_numbers

        i_number = number_list.index(number)
        assert i == ii * n_numbers + i_number, (i, ii, n_numbers, i_number, number_list)

        center_field = center[ii]
        mean_field = mean[ii]

        for k in keys + ["grid", "shape"]:
            if k == "number":
                continue
            assert center_field.metadata(k) == field.metadata(k), (
                k,
                center_field.metadata(k),
                field.metadata(k),
            )
            assert mean_field.metadata(k) == field.metadata(k), (
                k,
                mean_field.metadata(k),
                field.metadata(k),
            )

        e = field.to_numpy()
        m = mean_field.to_numpy()
        c = center_field.to_numpy()
        assert m.shape == c.shape, (m.shape, c.shape)

        FORCED_POSITIVE = [
            "q",
            "cp",
            "lsp",
            "tp",
        ]  # add "swl4", "swl3", "swl2", "swl1", "swl0", and more ?
        #################################
        # Actual computation happens here
        x = c - m + e
        if param in FORCED_POSITIVE:
            warnings.warn(f"Clipping {param} to be positive")
            x = np.maximum(x, 0)
        #################################

        assert x.shape == e.shape, (x.shape, e.shape)

        check_data_values(x, name=param)
        out.write(x, template=field)

    out.close()

    ds = load_source("file", path)
    assert_is_fieldset(ds)
    # save a reference to the tmp file so it is deleted
    # only when the dataset is not used anymore
    ds._tmp = tmp

    assert len(ds) == len(ensembles), (len(ds), len(ensembles))

    return ds


execute = ensembles_perturbations

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """

    common: &common
        name: mars
        # marser is the MARS containing ERA5 reanalysis dataset, avoid hitting the FDB server for nothing
        database: marser
        class: ea
        # date: $datetime_format($dates,%Y%m%d)
        # time: $datetime_format($dates,%H%M)
        date: 20221230/to/20230103
        time: '0000/1200'
        expver: '0001'
        grid: 20.0/20.0
        levtype: sfc
        param: [2t]
        # levtype: pl
        # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]

    config:
        ensembles: # the ensemble data has one additional dimension
          <<: *common
          stream: enda
          type: an
          number: [0, 1]
          # number: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        center: # the new center of the data
          <<: *common
          stream: oper
          type: an

        mean: # the previous center of the data
          <<: *common
          stream: enda
          type: em

    """
    )["config"]
    for k, v in config.items():
        print(k, v)

    for f in ensembles_perturbations(**config):
        print(f, f.to_numpy().mean())
