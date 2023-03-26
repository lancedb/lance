Install On Databricks Runtime
==========

The experience is based on Databricks Runtime 12.2, things may be different on other versions

Why it's different
----------
Usually we can install lance by just typing ``pip install pylance`` if you already have Python environment on your machine, but there is something spcecial in Databricks runtime. They of course have ``pyspark`` in their environment, but not as a normal package, it exists as a bunch of Python files in the path. The reason they do that should be they customized the ``pyspark`` a lot, and they don't want you accidently remove the customized one and install a vanilla ``pyspark`` instead.

But that leads to a problem, when your package depends or transitively depends on ``pyspark``, the ``pip`` command can't work properly. The ``pip`` in a DBR driver node is also customized, it will ignore your command of installing/unstalling/upgrading ``pyspark``, but seems will still check the restrictions when another package depends on ``pyspark``. That leads to error information similar to the below one.

.. code-block::
   !pip install --upgrade pandas
   Installing collected packages: pandas
   Attempting uninstall: pandas
   Found existing installation: pandas 1.4.2
   Not uninstalling pandas at /databricks/python3/lib/python3.9/site-packages, outside environment /local_disk0/.ephemeral_nfs/envs/pythonEnv-2dcfb7be-235f-41a5-be07-de1a5d89af5b
   Can't uninstall 'pandas'. No files were found to uninstall.
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
   petastorm 0.12.1 requires pyspark>=2.1.0, which is not installed.
   databricks-feature-store 0.10.0 requires pyspark<4,>=3.1.2, which is not installed.

A workable way so far
----------
The ``Libraries`` tab of the cluster on Databricks web console seems use a differently customized version of ``pip`` that can install ``pylance`` as well as other packages that depends on ``pyspark``. Because it's a black box, we don't know what exactly it does when the packages have some conflicts with the preinstalled ones in DBR, to install ``pylance`` as well as the important dependencies like ``pyarrow`` and ``pandas`` is a good security measure. The recommended versions are in `pyproject.toml <https://github.com/eto-ai/lance/blob/main/python/pyproject.toml>`_.

TODO: add a pic
