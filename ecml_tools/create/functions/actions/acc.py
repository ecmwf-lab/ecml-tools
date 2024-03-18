# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
from copy import deepcopy

import climetlab as cml
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output
from climetlab.utils.availability import Availability

from ecml_tools.create.utils import to_datetime_list

DEBUG = True


class Accumulation:
    def __init__(self, out, param, date, time, number, stepping):
        self.out = out
        self.param = param
        self.date = date
        self.time = time
        self.number = number
        self.values = None
        self.seen = set()
        self.startStep = None
        self.endStep = None
        self.done = False
        self.stepping = stepping

    @property
    def key(self):
        return (self.param, self.date, self.time, self.number)


class AccumulationFromStart(Accumulation):
    def add(self, field, values):
        step = field.metadata("step")
        # if step not in self.steps:
        #     return

        assert not self.done, (self.key, step)
        assert step not in self.seen, (self.key, step)

        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")

        assert startStep == 0 or (startStep == endStep), (startStep, endStep, step)
        assert step == endStep, (startStep, endStep, step)

        if self.values is None:
            import numpy as np

            self.values = np.copy(values)
            self.startStep = 0
            self.endStep = endStep
            ready = False

        else:
            assert endStep != self.endStep, (self.endStep, endStep)

            if endStep > self.endStep:
                # assert endStep - self.endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = values - self.values
                self.endStep = endStep
            else:
                # assert self.endStep - endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = self.values - values

            ready = True

        self.seen.add(step)

        if ready:
            self.out.write(
                self.values,
                template=field,
                startStep=self.startStep,
                endStep=self.endStep,
            )
            self.values = None
            self.done = True


class AccumulationFromLastStep(Accumulation):
    def add(self, field, values):
        step = field.metadata("step")

        assert not self.done, (self.key, step)
        assert step not in self.seen, (self.key, step)

        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")

        assert endStep == step, (startStep, endStep, step)
        assert step not in self.seen, (self.key, step)

        assert endStep - startStep == self.stepping, (startStep, endStep)

        if self.startStep is None:
            self.startStep = startStep
        else:
            self.startStep = min(self.startStep, startStep)

        if self.endStep is None:
            self.endStep = endStep
        else:
            self.endStep = max(self.endStep, endStep)

        if self.values is None:
            import numpy as np

            self.values = np.zeros_like(values)

        self.values += values

        self.seen.add(step)

        if len(self.seen) == len(self.steps):
            self.out.write(
                self.values,
                template=field,
                startStep=self.startStep,
                endStep=self.endStep,
            )
            self.values = None
            self.done = True


def accumulations_from_start(dates, step1, step2):
    for valid_date in dates:
        base_date = valid_date - datetime.timedelta(hours=step2)

        yield (
            base_date.year * 10000 + base_date.month * 100 + base_date.day,
            base_date.hour * 100 + base_date.minute,
            step1,
        )
        yield (
            base_date.year * 10000 + base_date.month * 100 + base_date.day,
            base_date.hour * 100 + base_date.minute,
            step2,
        )


def accumulations_from_last_step(dates, step1, step2, frequency):
    for valid_date in dates:
        date1 = valid_date - datetime.timedelta(hours=step1 + frequency)

        for step in range(step1, step2, frequency):
            date = date1 + datetime.timedelta(hours=step)
            yield (
                date.year * 10000 + date.month * 100 + date.day,
                date.hour * 100 + date.minute,
                step,
            )


def identity(x):
    return x


def accumulations(
    dates,
    data_accumulation_period,
    user_accumulation_period,
    request,
    patch=identity,
):
    if not isinstance(user_accumulation_period, (list, tuple)):
        user_accumulation_period = (0, user_accumulation_period)

    assert len(user_accumulation_period) == 2, user_accumulation_period
    step1, step2 = user_accumulation_period
    assert step1 < step2, user_accumulation_period

    if data_accumulation_period == 0:
        mars_date_time_step = accumulations_from_start(dates, step1, step2)
    else:
        mars_date_time_step = accumulations_from_last_step(dates, step1, step2, data_accumulation_period)

    request = deepcopy(request)

    param = request["param"]
    if not isinstance(param, (list, tuple)):
        param = [param]

    for p in param:
        assert p in ["cp", "lsp", "tp"], p

    number = request.get("number", [0])
    assert isinstance(number, (list, tuple))

    stepping = data_accumulation_period

    type_ = request.get("type", "an")
    if type_ == "an":
        type_ = "fc"

    request.update({"type": type_, "levtype": "sfc"})

    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    requests = []

    AccumulationClass = AccumulationFromStart if data_accumulation_period == 0 else AccumulationFromLastStep

    accumulations = {}

    for date, time, step in mars_date_time_step:
        for p in param:
            for n in number:
                requests.append(
                    patch(
                        {
                            "param": p,
                            "date": date,
                            "time": time,
                            "step": step,
                            "number": n,
                        }
                    )
                )

                key = (p, date, time, n)
                if key not in accumulations:
                    accumulations[key] = AccumulationClass(
                        out,
                        stepping=stepping,
                        param=p,
                        date=date,
                        time=time,
                        number=number,
                    )

    compressed = Availability(requests)
    ds = cml.load_source("empty")
    for r in compressed.iterate():
        request.update(r)
        ds = ds + cml.load_source("mars", **request)

    for field in ds:
        print(field)
        key = (
            field.metadata("param"),
            field.metadata("date"),
            field.metadata("time"),
            field.metadata("number"),
        )
        values = field.values  # optimisation
        accumulations[key].add(field, values)

    for a in accumulations.values():
        assert a.done, (a.key, a.seen)

    out.close()

    ds = cml.load_source("file", path)

    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )
    ds._tmp = tmp

    return ds


if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
      class: od
      expver: '0001'
      grid: 20./20.
      levtype: sfc
      param: tp
    """
    )
    dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    print(dates)

    def scda(request):
        if request["time"] in (600, 1800):
            request["stream"] = "scda"
        else:
            request["stream"] = "oper"
        return request

    ds = accumulations(dates, 0, (0, 6), config, scda)
    print()
    for f in ds:
        print(f.valid_datetime())
