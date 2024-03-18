# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import warnings
from collections import defaultdict
from copy import deepcopy

import climetlab as cml
import numpy as np
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output
from climetlab.utils.availability import Availability

from ecml_tools.create.utils import to_datetime_list

DEBUG = True


class Accumulation:
    def __init__(self, out, /, param, date, time, number, step, frequency, **kwargs):
        self.out = out
        self.param = param
        self.date = date
        self.time = time
        self.steps = step
        self.number = number
        self.values = None
        self.seen = set()
        self.startStep = None
        self.endStep = None
        self.done = False
        self.frequency = frequency
        self._check = None

    @property
    def key(self):
        return (self.param, self.date, self.time, self.steps, self.number)

    def check(self, field):
        if self._check is None:
            self._check = field.as_mars()
            assert self.param == field.metadata("param"), (self.param, field.metadata("param"))
            assert self.date == field.metadata("date"), (self.date, field.metadata("date"))
            assert self.time == field.metadata("time"), (self.time, field.metadata("time"))
            assert self.number == field.metadata("number"), (self.number, field.metadata("number"))

            return

        mars = field.as_mars()
        keys1 = sorted(self._check.keys())
        keys2 = sorted(mars.keys())

        assert keys1 == keys2, (keys1, keys2)

        for k in keys1:
            if k not in ("step",):
                assert self._check[k] == mars[k], (k, self._check[k], mars[k])

    def write(self, template):

        assert self.startStep != self.endStep, (self.startStep, self.endStep)
        assert np.all(self.values >= 0), (np.amin(self.values), np.amax(self.values))

        self.out.write(
            self.values,
            template=template,
            stepType="accum",
            startStep=self.startStep,
            endStep=self.endStep,
        )
        self.values = None
        self.done = True

    def add(self, field, values):

        self.check(field)

        step = field.metadata("step")
        if step not in self.steps:
            return

        if not np.all(values >= 0):
            warnings.warn(f"Negative values for {field}: {np.amin(values)} {np.amax(values)}")

        assert not self.done, (self.key, step)
        assert step not in self.seen, (self.key, step)

        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")

        if self.buggy_steps and startStep == endStep:
            startStep = 0

        assert step == endStep, (startStep, endStep, step)

        self.compute(values, startStep, endStep)

        self.seen.add(step)

        if len(self.seen) == len(self.steps):
            self.write(template=field)


class AccumulationFromStart(Accumulation):
    buggy_steps = True

    def compute(self, values, startStep, endStep):

        assert startStep == 0, startStep

        if self.values is None:

            self.values = np.copy(values)
            self.startStep = 0
            self.endStep = endStep

        else:
            assert endStep != self.endStep, (self.endStep, endStep)

            if endStep > self.endStep:
                # assert endStep - self.endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = values - self.values
                self.startStep = self.endStep
                self.endStep = endStep
            else:
                # assert self.endStep - endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = self.values - values
                self.startStep = endStep

            if not np.all(self.values >= 0):
                warnings.warn(f"Negative values for {self.param}: {np.amin(self.values)} {np.amax(self.values)}")
                self.values = np.maximum(self.values, 0)

    @classmethod
    def mars_date_time_steps(cls, dates, step1, step2, frequency, base_times):
        assert frequency == 0, frequency
        assert base_times is None, base_times

        for valid_date in dates:
            base_date = valid_date - datetime.timedelta(hours=step2)

            yield (
                base_date.year * 10000 + base_date.month * 100 + base_date.day,
                base_date.hour * 100 + base_date.minute,
                (step1, step2),
            )


class AccumulationFromLastStep(Accumulation):
    buggy_steps = False

    def compute(self, values, startStep, endStep):

        assert endStep - startStep == self.frequency, (startStep, endStep, self.frequency)

        if self.startStep is None:
            self.startStep = startStep
        else:
            self.startStep = min(self.startStep, startStep)

        if self.endStep is None:
            self.endStep = endStep
        else:
            self.endStep = max(self.endStep, endStep)

        if self.values is None:
            self.values = np.zeros_like(values)

        self.values += values

    @classmethod
    def mars_date_time_steps(cls, dates, step1, step2, frequency, base_times):

        if base_times is None:
            base_times = [0, 6, 12, 18]

        base_times = [t // 100 if t > 100 else t for t in base_times]
        assert frequency

        for valid_date in dates:

            print(f"====> {valid_date=}")

            base_date = valid_date - datetime.timedelta(hours=step2)
            add_step = 0
            while base_date.hour not in base_times:
                # print(f'{base_date=}, {base_times=}, {add_step=} {frequency=}')
                base_date -= datetime.timedelta(hours=frequency)
                add_step += frequency

            steps = []
            for step in range(step1 + frequency, step2 + frequency, frequency):
                steps.append(step + add_step)

            yield (
                base_date.year * 10000 + base_date.month * 100 + base_date.day,
                base_date.hour * 100 + base_date.minute,
                tuple(steps),
            )


def identity(x):
    return x


def compute_accumulations(
    dates,
    request,
    user_accumulation_period=6,
    data_accumulation_period=None,
    patch=identity,
    base_times=None,
):
    if not isinstance(user_accumulation_period, (list, tuple)):
        user_accumulation_period = (0, user_accumulation_period)

    assert len(user_accumulation_period) == 2, user_accumulation_period
    step1, step2 = user_accumulation_period
    assert step1 < step2, user_accumulation_period

    AccumulationClass = AccumulationFromStart if data_accumulation_period in (0, None) else AccumulationFromLastStep

    mars_date_time_steps = AccumulationClass.mars_date_time_steps(
        dates,
        step1,
        step2,
        data_accumulation_period,
        base_times,
    )

    request = deepcopy(request)

    param = request["param"]
    if not isinstance(param, (list, tuple)):
        param = [param]

    for p in param:
        assert p in ["cp", "lsp", "tp", "sf"], p

    number = request.get("number", [0])
    assert isinstance(number, (list, tuple))

    frequency = data_accumulation_period

    type_ = request.get("type", "an")
    if type_ == "an":
        type_ = "fc"

    request.update({"type": type_, "levtype": "sfc"})

    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    requests = []

    accumulations = {}

    for date, time, steps in mars_date_time_steps:
        for p in param:
            for n in number:
                requests.append(
                    patch(
                        {
                            "param": p,
                            "date": date,
                            "time": time,
                            "step": sorted(steps),
                            "number": n,
                        }
                    )
                )

    compressed = Availability(requests)
    ds = cml.load_source("empty")
    for r in compressed.iterate():
        request.update(r)
        ds = ds + cml.load_source("mars", **request)

    accumulations = defaultdict(list)
    for a in [AccumulationClass(out, frequency=frequency, **r) for r in requests]:
        for s in a.steps:
            accumulations[(a.param, a.date, a.time, s, a.number)].append(a)

    for field in ds:
        key = (
            field.metadata("param"),
            field.metadata("date"),
            field.metadata("time"),
            field.metadata("step"),
            field.metadata("number"),
        )
        values = field.values  # optimisation
        for a in accumulations[key]:
            a.add(field, values)

    for acc in accumulations.values():
        for a in acc:
            assert a.done, (a.key, a.seen, a.steps)

    out.close()

    ds = cml.load_source("file", path)

    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )
    ds._tmp = tmp

    return ds


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def normalise_time_to_hours(r):
    r = deepcopy(r)
    if "time" not in r:
        return r

    times = []
    for t in to_list(r["time"]):
        assert len(t) == 4, r
        assert t.endswith("00"), r
        times.append(int(t) // 100)
    r["time"] = tuple(times)
    return r


def normalise_number(r):
    if "number" not in r:
        return r
    number = r["number"]
    number = to_list(number)

    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    r["number"] = number
    return r


def scda(request):
    if request["time"] in (6, 18, 600, 1800):
        request["stream"] = "scda"
    else:
        request["stream"] = "oper"
    return request


def accumulations(context, dates, **request):
    to_list(request["param"])
    class_ = request.get("class", "od")
    stream = request.get("stream", "oper")

    user_accumulation_period = request.get("accumulation_period", 6)

    KWARGS = {
        ("od", "oper"): dict(patch=scda),
        ("ea", "oper"): dict(data_accumulation_period=1, base_times=(6, 18)),
        ("ea", "enda"): dict(data_accumulation_period=3, base_times=(6, 18)),
    }

    kwargs = KWARGS.get((class_, stream), {})

    return compute_accumulations(
        dates,
        request,
        user_accumulation_period=user_accumulation_period,
        **kwargs,
    )


execute = accumulations

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
      class: ea
      expver: '0001'
      grid: 20./20.
      levtype: sfc
#      number: [0, 1]
#      stream: enda
      param: [cp, tp]
#      accumulation_period: 6h
    """
    )
    dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in accumulations(None, dates, **config):
        print(f, f.to_numpy().mean())
