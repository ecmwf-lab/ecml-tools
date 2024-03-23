The ``mars`` source will retrieve the data from the ECMWF MARS archive.
For that, you need to have an ECMWF account and build your dataset on
one of the Centre's computers, or use the ``ecmwfapi`` Python package.

The `yaml` block can contain any keys that following the `MARS language
specification`_, with the exception of the ``date``, ``time``` and
``step``.

The missing keys will be filled with the default values, as defined in
the MARS language specification.

.. code:: yaml

   mars:
       levtype: sfc
       param: [2t, msl]

.. _mars language specification: https://confluence.ecmwf.int/display/UDOC/MARS+user+documentation
