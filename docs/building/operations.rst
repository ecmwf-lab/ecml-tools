.. _dataset-operations:

############
 Operations
############

******
 join
******

The join is the process of combining several sources data. Each
source is expected to provide different variables at the same dates.

.. code-block:: yaml

    input:
        join:
            - source1
            - source2
            - ...


********
 concat
********

The concatenation is the process of combining different sets of
operation that handle different dates. This is typically used to
build a dataset that spans several years, when the several sources
are involved, each providing a different period.

.. literalinclude:: concat.yaml
    :language: yaml


******
 pipe
******

The pipe is the process of transforming fields using filters. The
first step of a pipe is typically a source, a join or another pipe.
The following steps are filters.


.. code-block:: yaml

    input:
        pipe:
            - source
            - filter1
            - filter2
            - ...

