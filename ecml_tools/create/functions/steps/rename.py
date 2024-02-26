# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from climetlab.indexing.fieldset import FieldArray


class RenamedField:
    def __init__(self, field, what, renaming):
        self.field = field
        self.what = what
        self.renaming = renaming

    def metadata(self, key):
        value = self.field.metadata(key)
        if key == self.what:
            return self.renaming.get(value, value)
        return value

    def __getattr__(self, name):
        return getattr(self.field, name)


def execute(context, input, what="param", **kwargs):
    return FieldArray([RenamedField(fs, what, kwargs) for fs in input])
