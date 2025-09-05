# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import sys

from .lance import _lance_tools_cli


def lance_tools_cli():
    _lance_tools_cli(sys.argv)
