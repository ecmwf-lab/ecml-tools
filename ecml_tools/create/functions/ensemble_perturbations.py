# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import climetlab as cml
import numpy as np
import tqdm
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
    n_ensembles = len(normalise_number(ensembles["number"]))

    ensembles = cml.load_source(**ensembles)
    center = cml.load_source(**center)
    mean = cml.load_source(**mean)

    assert len(mean) * n_ensembles == len(ensembles), (
        len(mean),
        n_ensembles,
        len(ensembles),
    )
    assert len(center) * n_ensembles == len(ensembles), (
        len(center),
        n_ensembles,
        len(ensembles),
    )

    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    keys = ["param", "level", "valid_datetime", "number", "date", "time", "step"]

    ensembles_coords = ensembles.unique_values(*keys)
    center_coords = center.unique_values(*keys)
    mean_coords = mean.unique_values(*keys)

    for k in keys:
        if k == "number":
            assert len(mean_coords[k]) == 1
            assert len(center_coords[k]) == 1
            assert len(ensembles_coords[k]) == n_ensembles
            continue
        assert center_coords[k] == ensembles_coords[k], (
            k,
            center_coords,
            ensembles_coords,
        )
        assert center_coords[k] == mean_coords[k], (k, center_coords, mean_coords)

    for field in tqdm.tqdm(center):
        param = field.metadata("param")
        selection = dict(
            valid_datetime=field.metadata("valid_datetime"),
            param=field.metadata("param"),
            level=field.metadata("level"),
            date=field.metadata("date"),
            time=field.metadata("time"),
            step=field.metadata("step"),
        )
        mean_field = get_unique_field(mean, selection)

        m = mean_field.to_numpy()
        c = field.to_numpy()
        assert m.shape == c.shape, (m.shape, c.shape)

        for number in ensembles_coords["number"]:
            ensembles_field = get_unique_field(ensembles.sel(number=number), selection)
            e = ensembles_field.to_numpy()
            assert c.shape == e.shape, (c.shape, e.shape)

            x = c + m - e
            if param == "q":
                print("Clipping q")
                x = np.max(x, 0)

            check_data_values(x, name=param)
            out.write(x, template=ensembles_field)

    out.close()

    ds = cml.load_source("file", path)
    assert len(ds) == len(ensembles), (len(ds), len(ensembles))
    ds._tmp = tmp

    assert len(mean) * n_ensembles == len(ensembles)
    assert len(center) * n_ensembles == len(ensembles)

    final_coords = ds.unique_values(*keys)
    assert len(final_coords["number"]) == n_ensembles, final_coords
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
