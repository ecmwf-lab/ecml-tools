# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import os
import re
import warnings

import numpy as np
import tqdm

LOG = logging.getLogger(__name__)


def compute_directory_size(path):
    if not os.path.isdir(path):
        return None
    size = 0
    n = 0
    for dirpath, _, filenames in tqdm.tqdm(
        os.walk(path), desc="Computing size", leave=False
    ):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            size += os.path.getsize(file_path)
            n += 1
    return size, n


class DatasetName:
    def __init__(
        self,
        name,
        resolution=None,
        start_date=None,
        end_date=None,
        frequency=None,
    ):
        self.name = name
        self.parsed = self._parse(name)

        self.messages = []

        self.check_parsed()
        self.check_resolution(resolution)
        self.check_frequency(frequency)
        self.check_start_date(start_date)
        self.check_end_date(end_date)

        if self.messages:
            self.messages.append(
                f"{self} is parsed as :"
                + "/".join(f"{k}={v}" for k, v in self.parsed.items())
            )

    @property
    def is_valid(self):
        return not self.messages

    @property
    def error_message(self):
        out = " And ".join(self.messages)
        if out:
            out = out[0].upper() + out[1:]
        return out

    def raise_if_not_valid(self, print=print):
        if not self.is_valid:
            for m in self.messages:
                print(m)
            raise ValueError(self.error_message)

    def _parse(self, name):
        pattern = r"^(\w+)-(\w+)-(\w+)-(\w+)-(\w\w\w\w)-(\w+)-(\w+)-([\d\-]+)-(\d+h)-v(\d+)-?(.*)$"
        match = re.match(pattern, name)

        parsed = {}
        if match:
            keys = [
                "use_case",
                "class_",
                "type_",
                "stream",
                "expver",
                "source",
                "resolution",
                "period",
                "frequency",
                "version",
                "additional",
            ]
            parsed = {k: v for k, v in zip(keys, match.groups())}

            period = parsed["period"].split("-")
            assert len(period) in (1, 2), (name, period)
            parsed["start_date"] = period[0]
            if len(period) == 1:
                parsed["end_date"] = period[0]
            if len(period) == 2:
                parsed["end_date"] = period[1]

        return parsed

    def __str__(self):
        return self.name

    def check_parsed(self):
        if not self.parsed:
            self.messages.append(
                (
                    f"the dataset name {self} does not follow naming convention. "
                    "See here for details: "
                    "https://confluence.ecmwf.int/display/DWF/Datasets+available+as+zarr"
                )
            )

    def check_resolution(self, resolution):
        if (
            self.parsed.get("resolution")
            and self.parsed["resolution"][0] not in "0123456789on"
        ):
            self.messages.append(
                (
                    f"the resolution {self.parsed['resolution'] } should start "
                    f"with a number or 'o' or 'n' in the dataset name {self}."
                )
            )

        if resolution is None:
            return
        resolution_str = str(resolution).replace(".", "p").lower()
        self._check_missing("resolution", resolution_str)
        self._check_mismatch("resolution", resolution_str)

    def check_frequency(self, frequency):
        if frequency is None:
            return
        frequency_str = f"{frequency}h"
        self._check_missing("frequency", frequency_str)
        self._check_mismatch("frequency", frequency_str)

    def check_start_date(self, start_date):
        if start_date is None:
            return
        start_date_str = str(start_date.year)
        self._check_missing("first date", start_date_str)
        self._check_mismatch("start_date", start_date_str)

    def check_end_date(self, end_date):
        if end_date is None:
            return
        end_date_str = str(end_date.year)
        self._check_missing("end_date", end_date_str)
        self._check_mismatch("end_date", end_date_str)

    def _check_missing(self, key, value):
        if value not in self.name:
            self.messages.append(
                (f"the {key} is {value}, but is missing in {self.name}.")
            )

    def _check_mismatch(self, key, value):
        if self.parsed.get(key) and self.parsed[key] != value:
            self.messages.append(
                (f"the {key} is {value}, but is {self.parsed[key]} in {self.name}.")
            )


class StatisticsValueError(ValueError):
    pass


def check_data_values(arr, *, name: str, log=[]):
    min, max = arr.min(), arr.max()
    assert not (np.isnan(arr).any()), (name, min, max, *log)

    if min == 9999.0:
        warnings.warn(f"Min value 9999 for {name}")
    if max == 9999.0:
        warnings.warn(f"Max value 9999 for {name}")

    in_0_1 = dict(minimum=0, maximum=1)
    limits = {
        "lsm": in_0_1,
        "insolation": in_0_1,
        "2t": dict(minimum=173.15, maximum=373.15),
    }

    if name in limits:
        if min < limits[name]["minimum"]:
            raise StatisticsValueError(
                (
                    f"For {name}: minimum value in the data is {min}. "
                    "Not in acceptable range [{limits[name]['minimum']} ; {limits[name]['maximum']}]"
                )
            )
        if max > limits[name]["maximum"]:
            raise StatisticsValueError(
                (
                    f"For {name}: maximum value in the data is {max}. "
                    "Not in acceptable range [{limits[name]['minimum']} ; {limits[name]['maximum']}]"
                )
            )


def check_stats(minimum, maximum, mean, msg, **kwargs):
    tolerance = (abs(minimum) + abs(maximum)) * 0.01
    if (mean - minimum < -tolerance) or (mean - minimum < -tolerance):
        raise StatisticsValueError(
            f"Mean is not in min/max interval{msg} : we should have {minimum} <= {mean} <= {maximum}"
        )
