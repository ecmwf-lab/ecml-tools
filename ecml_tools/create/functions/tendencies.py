# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
from copy import deepcopy

from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output

from ecml_tools.create.functions import assert_is_fieldset
from ecml_tools.create.utils import to_datetime_list


def _date_to_datetime(d):
    if isinstance(d, (list, tuple)):
        return [_date_to_datetime(x) for x in d]
    return datetime.datetime.fromisoformat(d)


def normalise_time_delta(t):
    if isinstance(t, datetime.timedelta):
        assert t == datetime.timedelta(hours=t.hours), t

    assert t.endswith("h"), t

    t = int(t[:-1])
    t = datetime.timedelta(hours=t)
    return t


def tendencies(dates, time_increment, **kwargs):
    for d in dates:
        assert isinstance(d, datetime.datetime), (type(d), d)

    time_increment = normalise_time_delta(time_increment)

    assert len(kwargs) == 1, kwargs
    assert "function" in kwargs, kwargs
    func_kwargs = deepcopy(kwargs["function"])
    assert func_kwargs.pop("name") == "mars", kwargs
    from ecml_tools.create.functions.mars import execute as mars

    current_dates = [d.isoformat() for d in dates]
    shifted_dates = [(d - time_increment).isoformat() for d in dates]
    all_dates = sorted(list(set(current_dates + shifted_dates)))

    padded_source = mars(dates=all_dates, **func_kwargs)
    assert_is_fieldset(padded_source)
    assert len(padded_source)

    dates_in_data = padded_source.unique_values("valid_datetime")["valid_datetime"]
    print(dates_in_data)
    for d in current_dates:
        assert d in dates_in_data, d
    for d in shifted_dates:
        assert d in dates_in_data, d

    keys = ["valid_datetime", "date", "time", "step", "param", "level", "number"]
    current = padded_source.sel(valid_datetime=current_dates).order_by(*keys)
    before = padded_source.sel(valid_datetime=shifted_dates).order_by(*keys)

    assert len(current), (current, current_dates)

    assert len(current) == len(before), (
        len(current),
        len(before),
        time_increment,
        len(dates),
    )

    # prepare output tmp file so we can read it back
    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    for field, b_field in zip(current, before):
        for k in ["param", "level", "number", "grid", "shape"]:
            assert field.metadata(k) == b_field.metadata(k), (
                k,
                field.metadata(k),
                b_field.metadata(k),
            )
        c = field.to_numpy()
        b = b_field.to_numpy()
        assert c.shape == b.shape, (c.shape, b.shape)

        ################
        # Actual computation happens here
        x = c - b
        ################

        assert x.shape == c.shape, c.shape
        print(
            "computing data for",
            field.metadata("valid_datetime"),
            "=",
            field,
            "-",
            b_field,
        )
        out.write(x, template=field)

    out.close()

    from climetlab import load_source

    ds = load_source("file", path)
    assert_is_fieldset(ds)
    # save a reference to the tmp file so it is deleted
    # only when the dataset is not used anymore
    ds._tmp = tmp

    len(ds)
    len(padded_source)
    assert len(ds) == len(current), (len(ds), len(current))

    return ds


execute = tendencies

if __name__ == "__main__":
    import yaml

    #    config = yaml.safe_load(
    #        """
    #
    #    config:
    #        padded_source:
    #          name: mars
    #          # marser is the MARS containing ERA5 reanalysis dataset, avoid hitting the FDB server for nothing
    #          database: marser
    #          class: ea
    #          # date: $datetime_format($dates,%Y%m%d)
    #          # time: $datetime_format($dates,%H%M)
    #          date: 20221230/to/20230103
    #          time: '0000/1200'
    #          expver: '0001'
    #          grid: 20.0/20.0
    #          levtype: sfc
    #          param: [2t]
    #          # levtype: pl
    #          # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]
    #
    #          # number: [0, 1]
    #
    #        time_increment: 12h
    #
    #        dates: [2022-12-31 00:00, 2022-12-31 12:00, 2023-01-01 00:00, 2023-01-01 12:00, 2023-01-02 00:00, 2023-01-02 12:00]
    #    """
    #    )["config"]
    config = yaml.safe_load(
        """

    config:
        # name: tendencies
        # dates: $dates
        time_increment: 12h
        function:
          name: mars
          database: marser
          class: ea
          # date: computed automatically
          # time: computed automatically
          expver: "0001"
          grid: 20.0/20.0
          levtype: sfc
          param: [2t]
          # levtype: pl
          # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]
          # number: [0, 1]
          #  name: mars
          #  database: marser
          #  class: ea
          #  expver: '0001'
          #  grid: 20.0/20.0
          #  levtype: sfc
          #  param: [2t]
          #  # levtype: pl
          #  # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]
          #  # number: [0, 1]
    """
    )["config"]

    dates = yaml.safe_load(
        "[2022-12-30 12:00, 2022-12-31 00:00, 2022-12-31 12:00, 2023-01-01 00:00, 2023-01-01 12:00]"
    )
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in tendencies(dates, **config):
        print(f, f.to_numpy().mean())
