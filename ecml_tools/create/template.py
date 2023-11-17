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

from ecml_tools.create.utils import to_datetime

LOG = logging.getLogger(__name__)


def substitute(x, vars=None, ignore_missing=False):
    """Recursively substitute environment variables and dict values in a nested list ot dict of string.
    substitution is performed using the environment var (if UPPERCASE) or the input dictionary.


    >>> substitute({'bar': '$bar'}, {'bar': '43'})
    {'bar': '43'}

    >>> substitute({'bar': '$BAR'}, {'BAR': '43'})
    Traceback (most recent call last):
        ...
    KeyError: 'BAR'

    >>> substitute({'bar': '$BAR'}, ignore_missing=True)
    {'bar': '$BAR'}

    >>> os.environ["BAR"] = "42"
    >>> substitute({'bar': '$BAR'})
    {'bar': '42'}

    >>> substitute('$bar', {'bar': '43'})
    '43'

    >>> substitute('$hdates_from_date($date, 2015, 2018)', {'date': '2023-05-12'})
    '2015-05-12/2016-05-12/2017-05-12/2018-05-12'

    """
    if vars is None:
        vars = {}

    assert isinstance(vars, dict), vars

    if isinstance(x, (tuple, list)):
        return [substitute(y, vars, ignore_missing=ignore_missing) for y in x]

    if isinstance(x, dict):
        return {
            k: substitute(v, vars, ignore_missing=ignore_missing) for k, v in x.items()
        }

    if isinstance(x, str):
        if "$" not in x:
            return x

        lst = []

        for i, bit in enumerate(re.split(r"(\$(\w+)(\([^\)]*\))?)", x)):
            if bit is None:
                continue
            assert isinstance(bit, str), (bit, type(bit), x, type(x))

            i %= 4
            if i in [2, 3]:
                continue
            if i == 1:
                try:
                    if "(" in bit:
                        # substitute by a function
                        FUNCTIONS = dict(hdates_from_date=hdates_from_date)

                        pattern = r"\$(\w+)\(([^)]*)\)"
                        match = re.match(pattern, bit)
                        assert match, bit

                        function_name = match.group(1)
                        params = [p.strip() for p in match.group(2).split(",")]
                        params = [
                            substitute(p, vars, ignore_missing=ignore_missing)
                            for p in params
                        ]

                        bit = FUNCTIONS[function_name](*params)

                    elif bit.upper() == bit:
                        # substitute by the var env if $UPPERCASE
                        bit = os.environ[bit[1:]]
                    else:
                        # substitute by the value in the 'vars' dict
                        bit = vars[bit[1:]]
                except KeyError as e:
                    if not ignore_missing:
                        raise e

            if bit != x:
                bit = substitute(bit, vars, ignore_missing=ignore_missing)

            lst.append(bit)

        lst = [_ for _ in lst if _ != ""]
        if len(lst) == 1:
            return lst[0]

        out = []
        for elt in lst:
            # if isinstance(elt, str):
            #    elt = [elt]
            assert isinstance(elt, (list, tuple)), elt
            out += elt
        return out

    return x


def hdates_from_date(date, start_year, end_year):
    """
    Returns a list of dates in the format '%Y%m%d' between start_year and end_year (inclusive),
    with the year of the input date.

    Args:
        date (str or datetime): The input date.
        start_year (int): The start year.
        end_year (int): The end year.

    Returns:
        List[str]: A list of dates in the format '%Y%m%d'.
    """
    if not str(start_year).isdigit():
        raise ValueError(f"start_year must be an int: {start_year}")
    if not str(end_year).isdigit():
        raise ValueError(f"end_year must be an int: {end_year}")
    start_year = int(start_year)
    end_year = int(end_year)

    if isinstance(date, (list, tuple)):
        if len(date) != 1:
            raise NotImplementedError(f"{date} should have only one element.")
        date = date[0]

    date = to_datetime(date)
    assert not (date.hour or date.minute or date.second), date

    hdates = [date.replace(year=year) for year in range(start_year, end_year + 1)]
    return "/".join(d.strftime("%Y-%m-%d") for d in hdates)
