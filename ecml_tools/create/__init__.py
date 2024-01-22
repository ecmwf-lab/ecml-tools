# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os


class Creator:
    def __init__(
        self,
        path,
        config=None,
        cache=None,
        print=print,
        statistics_tmp=None,
        overwrite=False,
        **kwargs,
    ):
        self.path = path
        self.config = config
        self.cache = cache
        self.print = print
        self.statistics_tmp = statistics_tmp
        self.overwrite = overwrite
        # if kwargs:
        #    raise ValueError(f"Unknown arguments {kwargs}")

    def init(self, check_name=False):
        # check path
        _, ext = os.path.splitext(self.path)
        assert ext != "zarr", f"Unsupported extension={ext}"
        from .loaders import InitialiseLoader

        if self._path_readable() and not self.overwrite:
            raise Exception(
                f"{self.path} already exists. Use overwrite=True to overwrite."
            )

        cls = InitialiseLoader

        with self._cache_context():
            obj = cls.from_config(
                path=self.path,
                config=self.config,
                statistics_tmp=self.statistics_tmp,
                print=self.print,
            )
            obj.initialise(check_name=check_name)

    def load(self, parts=None):
        from .loaders import ContentLoader

        with self._cache_context():
            loader = ContentLoader.from_dataset_config(
                path=self.path, statistics_tmp=self.statistics_tmp, print=self.print
            )
            loader.load(parts=parts)

    def statistics(
        self,
        force=False,
        output=None,
    ):
        from .loaders import StatisticsLoader

        loader = StatisticsLoader.from_dataset(
            path=self.path,
            print=self.print,
            force=force,
            statistics_tmp=self.statistics_tmp,
            statistics_output=output,
            recompute=False,
        )
        loader.run()

    def recompute_statistics(
        self,
        start=None,
        end=None,
        force=False,
    ):
        from .loaders import StatisticsLoader

        loader = StatisticsLoader.from_dataset(
            path=self.path,
            print=self.print,
            force=force,
            statistics_tmp=self.statistics_tmp,
            statistics_start=start,
            statistics_end=end,
            recompute=True,
        )
        loader.run()

    def size(self):
        from .loaders import SizeLoader

        loader = SizeLoader.from_dataset(path=self.path, print=self.print)
        loader.add_total_size()

    def patch(self, **kwargs):
        from .patch import apply_patch

        apply_patch(self.path, **kwargs)

    def finalise(self, **kwargs):
        self.statistics(**kwargs)
        self.size()

    def create(self):
        self.init()
        self.load()
        self.finalise()

    def _cache_context(self):
        from .utils import cache_context

        return cache_context(self.cache)

    def _path_readable(self):
        import zarr

        try:
            zarr.open(self.path, "r")
            return True
        except zarr.errors.PathNotFoundError:
            return False
