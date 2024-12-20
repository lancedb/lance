# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import os
from typing import Optional

ENV_NAME_PYLANCE_LOGGING_LEVEL = "LANCE_LOG"


# Rust has 'trace' and Python does not so we map it to 'debug'
def get_python_log_level(rust_log_level: str) -> str:
    if rust_log_level.lower() == "trace":
        return "DEBUG"
    return rust_log_level


def get_log_level():
    lance_log_level = os.environ.get(ENV_NAME_PYLANCE_LOGGING_LEVEL, "INFO").upper()
    if lance_log_level == "":
        return "INFO"

    lance_log_level = [
        entry for entry in lance_log_level.split(",") if "=" not in entry
    ]
    if len(lance_log_level) > 0:
        return get_python_log_level(lance_log_level[0])
    else:
        return "INFO"


LOGGER = logging.getLogger("pylance")
LOGGER.setLevel(get_log_level())


def set_logger(
    file_path: Optional[str] = "pylance.log",
    name="pylance",
    level=logging.INFO,
    format_string=None,
    log_handler=None,
):
    global LOGGER
    if not format_string:
        format_string = "%(asctime)s %(name)s [%(levelname)s] %(filename)s:%(lineno)d %(funcName)s : %(message)s"  # noqa E501
    LOGGER = logging.getLogger(name)
    LOGGER.setLevel(level)
    lh = log_handler
    if lh is None:
        lh = logging.FileHandler(file_path)
    lh.setLevel(level)
    formatter = logging.Formatter(format_string)
    lh.setFormatter(formatter)
    LOGGER.addHandler(lh)
