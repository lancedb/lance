# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Configuration file for the Sphinx documentation builder.

import sys
import os

# -- Project information -----------------------------------------------------

project = "pylance"
copyright = "%Y, Lance Developer"
author = "Lance Developer"

sys.path.insert(0, os.path.abspath("../"))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    'sphinx.ext.autosummary',
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

autodoc_typehints = "signature"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
}
intersphinx_disabled_domains = ['std']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "collapse_navigation": False,   # Show all entries expanded
    "navigation_depth": 4,          # Show nested headings
    "titles_only": False            # Show both page titles and section titles
}

# -- doctest configuration ---------------------------------------------------

doctest_global_setup = """
import os
import shutil
from typing import Iterator

import lance
import pyarrow as pa
import numpy as np
import pandas as pd
"""

# Only test code examples in rst files
doctest_test_doctest_blocks = ""
