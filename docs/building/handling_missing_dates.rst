########################
 Handling missing dates
########################

Missing dates can be handled by specifying a list of dates in the
configuration file. The dates should be in the same format as the dates
in the time series. The missing dates will be filled ``np.nan`` values.

.. literalinclude:: missing_dates.yaml
   :language: yaml
