.. _data_management:

***************
Data Management
***************


Documentation of the code in *src.data_management*.

Here we clean a selected subset of the GATE's project original data. This
includes renaming variables, managing missing values, create dummy variables
from categorical ones, and generate new variables from existing ones.

To pre-process the data, we follow the implementation by Fairlie,
Karlan and Zinman :cite:`fairlie2015behind`. The original SAS code is
available `here`_.

.. _here: https://www.openicpsr.org/openicpsr/project/114561/version/V1/view?path=/openicpsr/114561/fcr:versions/V1/programfiles/crdata_v17.sas&type=file


``clean_data.py``
=====================

.. automodule:: src.data_management.clean_data
    :members:
