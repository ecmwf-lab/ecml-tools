# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging


LOG = logging.getLogger(__name__)


class DataRequest:
    @classmethod
    def from_zarr(cls, request):
        return ZarrRequest(request)

    def as_dict(self):
        return {
            "grid": self.grid,
            "area": self.area,
            "param_sfc": self.param_sfc,
            "param_level_pl": self.param_level_pl,
            "param_level_ml": self.param_level_ml,
            "param_step": self.param_step,
        }


class ZarrRequest(DataRequest):
    def __init__(self, request):
        self.request = request

    def select(self, variables):
        return Select(self, variables)

    @property
    def grid(self):
        return self.request["grid"]

    @property
    def area(self):
        return self.request["area"]

    @property
    def param_sfc(self):
        return self.request["param_level"].get("sfc", [])

    @property
    def param_level_pl(self):
        return self.request["param_level"].get("pl", [])

    @property
    def param_level_ml(self):
        return self.request["param_level"].get("ml", [])

    @property
    def param_step(self):
        return self.request.get("param_step", [])


class Select(DataRequest):
    def __init__(self, forward, variables):
        self.forward = forward
        self.variables = variables

    @property
    def grid(self):
        return self.forward.grid

    @property
    def area(self):
        return self.forward.area

    @property
    def param_sfc(self):
        return [x for x in self.forward.param_sfc if x in self.variables]

    @property
    def param_level_pl(self):
        return [
            x for x in self.forward.param_level_pl if f"{x[0]}_{x[1]}" in self.variables
        ]

    @property
    def param_level_ml(self):
        return [
            x for x in self.forward.param_level_ml if f"{x[0]}_{x[1]}" in self.variables
        ]

    @property
    def param_step(self):
        return [x for x in self.forward.param_step if x[0] in self.variables]
