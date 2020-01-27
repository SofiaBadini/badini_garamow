.. _introduction:


************
Introduction
************

Our project is concerned with the sensitivity results of OLS results to different imputation methods for missing values in an RCT framework.
For this purpose we base our analysis on the large RCT, the Growing America through Entrepreneurship (GATE) project that was implemented by xxx and evaluated by xxx.

.. _project_paths:

Project paths
=============

A variety of project paths are defined in the top-level wscript file. These are exported to header files in other languages. So in case you require different paths (e.g. if you have many different datasets, you may want to have one path to each of them), adjust them in the top-level wscript file.

The following is taken from the top-level wscript file.

.. literalinclude:: ../../wscript
    :start-after: out = "bld"
    :end-before:     # Convert the directories into Waf nodes


As should be evident from the similarity of the names, the paths follow the steps of the analysis in the :file:`src` directory:

    1. **data_management** → **OUT_DATA**
    2. **analysis** → **OUT_ANALYSIS**, **OUT_FIGURES**, **OUT_TABLES**
    3. **final** → **OUT_FINAL**

These will re-appear in automatically generated header files by calling the ``write_project_paths`` task generator (just use an output file with the correct extension for the language you need -- ``.py``, ``.r``, ``.m``, ``.do``)

By default, these header files are generated in the top-level build directory, i.e. ``bld``. The Python version defines a dictionary ``project_paths`` and a couple of convencience functions documented below. You can access these by adding a line::

    from bld.project_paths import XXX

at the top of you Python-scripts. Here is the documentation of the module:

    **bld.project_paths**

    .. automodule:: bld.project_paths
        :members:
