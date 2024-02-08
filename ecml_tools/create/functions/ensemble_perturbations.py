# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import warnings

import numpy as np
import tqdm
from climetlab import load_source
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output

from ecml_tools.create.check import check_data_values


def get_unique_field(ds, selection):
    ds = ds.sel(**selection)
    assert len(ds) == 1, (ds, selection)
    return ds[0]


def normalise_number(number):
    if isinstance(number, (tuple, list, int)):
        return number

    assert isinstance(number, str), (type(number), number)

    number = number.split("/")
    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    assert isinstance(number, list), (type(number), number)
    return number


def ensembles_perturbations(ensembles, center, mean, remapping={}, patches={}):
    number_list = normalise_number(ensembles["number"])
    n_numbers = len(number_list)

    keys = ["param", "level", "valid_datetime", "date", "time", "step", "number"]

    print(f"Retrieving ensemble data with {ensembles}")
    ensembles = load_source(**ensembles).order_by(*keys)
    print(f"Retrieving center data with {center}")
    center = load_source(**center).order_by(*keys)
    print(f"Retrieving mean data with {mean}")
    mean = load_source(**mean).order_by(*keys)

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
