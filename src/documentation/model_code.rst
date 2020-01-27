.. _model_code:

**********
Model Code
**********

Documentation of the code in *src.model_code*.


Impute missing observations
============================

``impute.py``
-------------

The python script ``impute`` creates different data sets based on the different imputation methods defined in ``model_specs``.
We use the **hot-deck** procedure and **weights** to account for missing observations.

.. automodule:: src.model_code.update
    :members:
