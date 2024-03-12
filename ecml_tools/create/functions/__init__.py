# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import os

from climetlab import load_source


class Mockup:
    write_directory = None
    read_directory = None

    def get_filename(self, r):
        import hashlib

        h = hashlib.md5(str(r).encode("utf8")).hexdigest()
        return h + ".copy"

    def write(self, ds, *args, **kwargs):
        if self.write_directory is None:
            return
        if not hasattr(ds, "path"):
            return

        path = os.path.join(self.write_directory, self.get_filename([args, kwargs]))
        print(f"Saving to {path} for {args}, {kwargs}")
        import shutil

        shutil.copy(ds.path, path)

    def load_source(self, *args, **kwargs):
        if self.read_directory is None:
            return None
        path = os.path.join(self.read_directory, self.get_filename([args, kwargs]))
        if os.path.exists(path):
            print(f"Loading path {path} for {args}, {kwargs}")
            return load_source("file", path)
        elif path.startswith("http"):
            import requests

            print(f"Loading url {path} for {args}, {kwargs}")
            try:
                return load_source("url", path)
            except requests.exceptions.HTTPError:
                pass
        return None


MOCKUP = Mockup()


def enable_save_mars(d):
    MOCKUP.write_directory = d


def disable_save_mars():
    MOCKUP.write_directory = None


def enable_read_mars(d):
    MOCKUP.read_directory = d


def disable_read_mars():
    MOCKUP.read_directory = None


if os.environ.get("MOCKUP_MARS_SAVE_REQUESTS"):
    enable_save_mars(os.environ.get("MOCKUP_MARS_SAVE_REQUESTS"))

if os.environ.get("MOCKUP_MARS_READ_REQUESTS"):
    enable_read_mars(os.environ.get("MOCKUP_MARS_READ_REQUESTS"))


def _load_source(*args, **kwargs):
    ds = MOCKUP.load_source(*args, **kwargs)
    if ds is None:
        ds = load_source(*args, **kwargs)
    MOCKUP.write(ds, *args, **kwargs)
    return ds


def assert_is_fieldset(obj):
    from climetlab.readers.grib.index import FieldSet

    assert isinstance(obj, FieldSet), type(obj)
