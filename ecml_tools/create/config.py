# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import logging
import os
from copy import deepcopy

import yaml
from climetlab.core.order import normalize_order_by

from .utils import load_json_or_yaml

LOG = logging.getLogger(__name__)


class DictObj(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictObj(value)
                continue
            if isinstance(value, list):
                self[key] = [
                    DictObj(item) if isinstance(item, dict) else item for item in value
                ]
                continue

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        self[attr] = value


def resolve_includes(config):
    if isinstance(config, list):
        return [resolve_includes(c) for c in config]
    if isinstance(config, dict):
        include = config.pop("<<", {})
        new = deepcopy(include)
        new.update(config)
        return {k: resolve_includes(v) for k, v in new.items()}
    return config


class Config(DictObj):
    def __init__(self, config):
        if isinstance(config, str):
            self.config_path = os.path.realpath(config)
            config = load_json_or_yaml(config)
        else:
            config = deepcopy(config)
        config = resolve_includes(config)
        super().__init__(config)


class OutputSpecs:
    def __init__(self, config, parent):
        self.config = config
        if "order_by" in config:
            assert isinstance(config.order_by, dict), config.order_by

        self.parent = parent

    @property
    def dtype(self):
        return self.config.dtype

    @property
    def order_by_as_list(self):
        # this is used when an ordered dict is not supported (e.g. zarr attributes)
        return [{k: v} for k, v in self.config.order_by.items()]

    def get_chunking(self, coords):
        user = deepcopy(self.config.chunking)
        chunks = []
        for k, v in coords.items():
            if k in user:
                chunks.append(user.pop(k))
            else:
                chunks.append(len(v))
        if user:
            raise ValueError(
                f"Unused chunking keys from config: {list(user.keys())}, not in known keys : {list(coords.keys())}"
            )
        return tuple(chunks)

    @property
    def append_axis(self):
        return self.config.append_axis

    @property
    def order_by(self):
        return self.config.order_by

    @property
    def remapping(self):
        return self.config.remapping

    @property
    def flatten_grid(self):
        return self.config.flatten_grid

    @property
    def statistics(self):
        return self.config.statistics


class LoadersConfig(Config):
    purpose = "undefined"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if "description" not in self:
            raise ValueError("Must provide a description in the config.")

        if "config_format_version" not in self:
            # Should be changed to 2
            self.config_format_version = 1
            print(
                f"Setting config_format_version={self.config_format_version} because it was not provided."
            )

        if self.config_format_version != 2:
            raise ValueError(
                "Config format has changed. Must provide config with format version >= 2."
            )

        if "dates" in self.output:
            raise ValueError("Obsolete: Dates should not be provided in output config.")

        # deprecated/obsolete
        if "order" in self.output:
            raise ValueError(
                f"Do not use 'order'. Use order_by instead. {list(self.keys())}"
            )
        if "loops" in self:
            raise ValueError(
                f"Do not use 'loops'. Use dates instead. {list(self.keys())}"
            )
        if "loop" in self:
            raise ValueError(
                f"Do not use 'loop'. Use dates instead. {list(self.keys())}"
            )

        if not isinstance(self.dates, dict):
            raise ValueError(f"Dates must be a dict. Got {self.dates}")

        self.normalise()

    def normalise(self):
        if isinstance(self.input, (tuple, list)):
            self.input = dict(concat=self.input)

        if "order_by" in self.output:
            self.output.order_by = normalize_order_by(self.output.order_by)

        self.output.chunking = self.output.get("chunking", {})
        self.output.dtype = self.output.get("dtype", "float32")

        self.reading_chunks = self.get("reading_chunks")
        assert "flatten_values" not in self.output
        assert "flatten_grid" in self.output, self.output

        # The axis along which we append new data
        # TODO: assume grid points can be 2d as well
        self.output.append_axis = 0

        assert "statistics" in self.output
        statistics_axis_name = self.output.statistics
        statistics_axis = -1
        for i, k in enumerate(self.output.order_by):
            if k == statistics_axis_name:
                statistics_axis = i

        assert (
            statistics_axis >= 0
        ), f"{self.output.statistics} not in {list(self.output.order_by.keys())}"

        self.statistics_names = self.output.order_by[statistics_axis_name]

        # TODO: consider 2D grid points
        self.statistics_axis = statistics_axis

    @classmethod
    def _get_first_key_if_dict(cls, x):
        if isinstance(x, str):
            return x
        return list(x.keys())[0]

    def check_dict_value_and_set(self, dic, key, value):
        if key in dic:
            if dic[key] == value:
                return
            raise ValueError(
                f"Cannot use {key}={dic[key]} with {self.purpose} purpose. Must use {value}."
            )
        print(f"Setting {key}={value} because purpose={self.purpose}")
        dic[key] = value

    def ensure_element_in_list(self, lst, elt, index):
        if elt in lst:
            assert lst[index] == elt
            return lst

        _lst = [self._get_first_key_if_dict(d) for d in lst]
        if elt in _lst:
            assert _lst[index] == elt
            return lst

        return lst[:index] + [elt] + lst[index:]


class UnknownPurposeConfig(LoadersConfig):
    purpose = "unknown"

    def normalise(self):
        self.output.flatten_grid = self.output.get("flatten_grid", False)
        self.output.ensemble_dimension = self.output.get("ensemble_dimension", False)
        super().normalise()  # must be called last


class AifsPurposeConfig(LoadersConfig):
    purpose = "aifs"

    def normalise(self):
        if "licence" not in self:
            self.licence = "CC-BY-4.0"
            print(f"❗ Setting licence={self.licence} because it was not provided.")
        if "copyright" not in self:
            self.copyright = "ecmwf"
            print(f"❗ Setting copyright={self.copyright} because it was not provided.")

        self.check_dict_value_and_set(self.output, "flatten_grid", True)
        self.check_dict_value_and_set(self.output, "ensemble_dimension", 2)

        assert isinstance(self.output.order_by, (list, tuple)), self.output.order_by
        self.output.order_by = self.ensure_element_in_list(
            self.output.order_by, "number", self.output.ensemble_dimension
        )

        order_by = self.output.order_by
        assert len(order_by) == 3, order_by
        assert self._get_first_key_if_dict(order_by[0]) == "valid_datetime", order_by
        assert self._get_first_key_if_dict(order_by[2]) == "number", order_by

        super().normalise()  # must be called last

    def get_serialisable_dict(self):
        return _prepare_serialisation(self)

    def get_variables_names(self):
        return self.output.order_by[self.output.statistics]


def _prepare_serialisation(o):
    if isinstance(o, dict):
        dic = {}
        for k, v in o.items():
            v = _prepare_serialisation(v)
            if k == "order_by" and isinstance(v, dict):
                # zarr attributes are saved with sort_keys=True
                # and ordered dict are reordered.
                # This is a problem for "order_by"
                # We ensure here that the order_by key contains
                # a list of dict
                v = [{kk: vv} for kk, vv in v.items()]
            dic[k] = v
        return dic

    if isinstance(o, (list, tuple)):
        return [_prepare_serialisation(v) for v in o]

    if o in (None, True, False):
        return o

    if isinstance(o, (str, int, float)):
        return o

    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

    return str(o)


CONFIGS = {
    None: UnknownPurposeConfig,
    "aifs": AifsPurposeConfig,
}


def loader_config(config):
    config = Config(config)
    obj = CONFIGS[config.get("purpose")](config)

    # yaml round trip to check that serialisation works as expected
    copy = obj.get_serialisable_dict()
    copy = yaml.load(yaml.dump(copy), Loader=yaml.SafeLoader)
    copy = Config(copy)
    copy = CONFIGS[config.get("purpose")](config)
    assert yaml.dump(obj) == yaml.dump(copy), (obj, copy)

    return copy


def build_output(*args, **kwargs):
    return OutputSpecs(*args, **kwargs)
