# Configuration file for the Sphinx documentation builder.

import sys
import os

# -- Project information -----------------------------------------------------

project = "Lance"
copyright = "%Y, Lance Developer"
author = "Lance Developer"

sys.path.insert(0, os.path.abspath("../python"))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
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
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "ray": ("https://docs.ray.io/en/latest/", None),
}

python_apigen_modules = {
    "lance": "api/python/",
}
object_description_options = [
    (
        "py:.*",
        dict(
            include_object_type_in_xref_tooltip=False,
            include_in_toc=False,
            include_fields_in_toc=False,
        ),
    ),
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_immaterial"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_favicon = "_static/favicon_64x64.png"
html_logo = "_static/high-res-icon.png"
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://github.com/lancedb/lance",
    "repo_url": "https://github.com/lancedb/lance",
    "repo_name": "Lance",
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        "content.tabs.link",
        "content.code.copy",
    ],
    "navigation_depth": 4,
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/jbms/sphinx-immaterial",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/pylance/",
        },
    ],
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
