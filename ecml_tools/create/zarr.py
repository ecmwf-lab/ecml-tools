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

import numpy as np

LOG = logging.getLogger(__name__)


def add_zarr_dataset(
    *,
    name,
    dtype=None,
    fill_value=np.nan,
    zarr_root,
    shape=None,
    array=None,
    overwrite=True,
    **kwargs,
):
    if dtype is None:
        assert array is not None, (name, shape, array, dtype, zarr_root, fill_value)
        dtype = array.dtype

    if shape is None:
        assert array is not None, (name, shape, array, dtype, zarr_root, fill_value)
        shape = array.shape
    else:
        assert array is None, (name, shape, array, dtype, zarr_root, fill_value)
        array = np.full(shape, fill_value, dtype=dtype)

    a = zarr_root.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        overwrite=overwrite,
        **kwargs,
    )
    a[...] = array
    return a


class ZarrRegistry:
    synchronizer_name = None  # to be defined in subclasses

    def __init__(self, path):
        assert self.synchronizer_name is not None, self.synchronizer_name

        import zarr

        assert isinstance(path, str), path
        self.zarr_path = path
        self.synchronizer = zarr.ProcessSynchronizer(self._synchronizer_path)

    @property
    def _synchronizer_path(self):
        return self.zarr_path + "-" + self.synchronizer_name + ".sync"

    def _open_write(self):
        import zarr

        return zarr.open(self.zarr_path, mode="r+", synchronizer=self.synchronizer)

    def _open_read(self, sync=True):
        import zarr

        if sync:
            return zarr.open(self.zarr_path, mode="r", synchronizer=self.synchronizer)
        else:
            return zarr.open(self.zarr_path, mode="r")

    def new_dataset(self, *args, **kwargs):
        z = self._open_write()
        zarr_root = z["_build"]
        add_zarr_dataset(*args, zarr_root=zarr_root, overwrite=True, **kwargs)

    def add_to_history(self, action, **kwargs):
        new = dict(
            action=action,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        new.update(kwargs)

        z = self._open_write()
        history = z.attrs.get("history", [])
        history.append(new)
        z.attrs["history"] = history


class ZarrStatisticsRegistry(ZarrRegistry):
    names = [
        "mean",
        "stdev",
        "minimum",
        "maximum",
        "sums",
        "squares",
        "count",
    ]
    build_names = [
        "minimum",
        "maximum",
        "sums",
        "squares",
        "count",
    ]
    synchronizer_name = "statistics"

    def __init__(self, path):
        super().__init__(path)

    def create(self):
        z = self._open_read()
        shape = z["data"].shape
        shape = (shape[0], shape[1])

        for name in self.build_names:
            if name == "count":
                self.new_dataset(name=name, shape=shape, fill_value=0, dtype=np.int64)
            else:
                self.new_dataset(
                    name=name, shape=shape, fill_value=np.nan, dtype=np.float64
                )
        self.add_to_history("statistics_initialised")

    def __setitem__(self, key, stats):
        z = self._open_write()

        LOG.info(f"Writting stats for {key}")
        for name in self.build_names:
            LOG.info(f"Writting stats for {key} {name} {stats[name].shape}")
            z["_build"][name][key] = stats[name]
        LOG.info(f"Written stats for {key}")

    def get_by_name(self, name):
        z = self._open_read()
        return z["_build"][name]


class ZarrBuiltRegistry(ZarrRegistry):
    name_lengths = "lengths"
    name_flags = "flags"
    lengths = None
    flags = None
    z = None
    synchronizer_name = "build"

    def get_slice_for(self, i):
        lengths = self.get_lengths()
        assert i >= 0 and i < len(lengths)

        start = sum(lengths[:i])
        stop = sum(lengths[: (i + 1)])
        return slice(start, stop)

    def get_lengths(self):
        z = self._open_read()
        return list(z["_build"][self.name_lengths][:])

    def get_flags(self, **kwargs):
        z = self._open_read(**kwargs)
        LOG.info(list(z["_build"][self.name_flags][:]))
        return list(z["_build"][self.name_flags][:])

    def get_flag(self, i):
        z = self._open_read()
        return z["_build"][self.name_flags][i]

    def set_flag(self, i, value=True):
        z = self._open_write()
        z.attrs["latest_write_timestamp"] = datetime.datetime.utcnow().isoformat()
        z["_build"][self.name_flags][i] = value

    def create(self, lengths, overwrite=False):
        self.new_dataset(name=self.name_lengths, array=np.array(lengths, dtype="i4"))
        self.new_dataset(
            name=self.name_flags, array=np.array([False] * len(lengths), dtype=bool)
        )
        self.add_to_history("initialised")

    def reset(self, lengths):
        return self.create(lengths, overwrite=True)

    def add_provenance(self, name):
        from ecml_tools.provenance import gather_provenance_info

        z = self._open_write()
        z.attrs[name] = gather_provenance_info()
