# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os
from contextlib import contextmanager


class EntryPoint:
    @classmethod
    def initialise(
        cls,
        path,
        config,
        force=False,
        no_check_name=True,
        cache_dir=None,
        print=print,
    ):
        # check path
        _, ext = os.path.splitext(path)
        assert ext != "zarr", f"Unsupported extension={ext}"
        from .loaders import InitialiseCreator

        cls = InitialiseCreator

        if cls.already_exists(path) and not force:
            raise Exception(f"{path} already exists. Use --force to overwrite.")

        with cache_context(cache_dir):
            obj = cls.from_config(partial=True, path=path, config=config, print=print)
            obj.initialise(check_name=not no_check_name)

    @classmethod
    def load(
        cls,
        path,
        parts=None,
        cache_dir=None,
        print=print,
    ):
        from .loaders import LoadCreator

        with cache_context(cache_dir):
            loader = LoadCreator.from_dataset(path=path, print=print)
            loader.load(parts=parts)

    @classmethod
    def add_total_size(cls, path):
        from .loaders import SizeCreator

        loader = SizeCreator.from_dataset(path=path, print=print)
        loader.add_total_size()


def cache_context(cache_dir):
    @contextmanager
    def no_cache_context():
        yield

    if cache_dir is None:
        return no_cache_context()

    from climetlab import settings

    os.makedirs(cache_dir, exist_ok=True)
    return settings.temporary("cache-directory", cache_dir)
