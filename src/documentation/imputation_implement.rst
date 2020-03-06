.. _imputation_implement:

**************************
Imputation implementation
**************************

Documentation of the code in *src.imputation_implement*.

Apply the imputation methods
=============================


``impute.py``
--------------

The python script ``impute`` applies the imputation methods which are defined in
``imputation_method.py``. These methods applied to the original data produce
data frames with full entries. These full data frames are then used to
run the OLS regression in ``itt_analysis.py'').

.. automodule:: src.imputation_implement.impute
    :members:
