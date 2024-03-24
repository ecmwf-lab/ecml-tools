.. _selecting-other:

##################
 Other operations
##################

.. warning:: The operations described in this section are do not check that their inputs are compatible.


*****
 zip
*****

The `zip` operation is used to combine multiple datasets into a single dataset.

.. code:: python

   ds = open_dataset(zip=[dataset1, dataset2, ...])

   # This will return tuples

   print(ds[0])

   print(ds[3, 4])



This operation is identical to the Python's :py:func:`zip` function.

*******
 chain
*******

.. code:: python

   ds = open_dataset(chain=[dataset1, dataset2, ...])


The `chain` operation is used to combine multiple datasets into a single dataset.
The datasets are combined by concatenating the data arrays along the first dimension (dates).
This is similar to the :ref:`concat` operation, but no check are done to see if the datasets are compatible,
this means that the shape of the arrays returned when iterating or indexing may be different.

This operation is identical to the Python's :py:func:`itertools.chain` function.


********
 shuffle
********

.. code:: python

   ds = open_dataset(dataset, shuffle=True)
