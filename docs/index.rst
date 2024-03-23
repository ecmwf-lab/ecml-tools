####################################
 Welcome to Anemoi's documentation!
####################################

.. warning::

   This documentation is work in progress. It is not yet ready.
   Currently, the documentation is based on the one from the ecml-tools_
   project, which will be merged into Anemoi.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. Anemoi provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

-  :doc:`overview`
-  :doc:`installing`
-  :doc:`firststeps`
-  :doc:`examples`

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installing
   firststeps
   examples

**Using training datasets**

-  :doc:`using/introduction`
-  :doc:`using/options`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   using/introduction
   using/options

**Building training datasets**

-  :doc:`building/introduction`
-  :doc:`building/sources`
-  :doc:`building/filters`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Building datasets

   building/introduction
   building/sources
   building/filters

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. _ecml-tools: https://github.com/ecmwf-lab/ecml-tools

.. _zarr: https://zarr.readthedocs.io/
