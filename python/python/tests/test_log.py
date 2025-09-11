# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import os
import re
import subprocess
import sys
from unittest import mock

import pytest
from lance.log import ENV_NAME_PYLANCE_LOGGING_LEVEL, LOGGER, get_log_level, set_logger


@pytest.fixture(autouse=True)
def teardown_logger():
    yield
    while LOGGER.handlers:
        LOGGER.handlers.pop()


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("DEBUG", "DEBUG"),
        ("INFO", "INFO"),
        ("WARNING", "WARNING"),
        ("DEBUG,INFO", "DEBUG"),
        ("", "INFO"),
        ("lance-core=debug,WARNING", "WARNING"),
        ("DEBUG,lance-core=WARNING", "DEBUG"),
    ],
)
def test_get_log_level(env_value, expected):
    with mock.patch.dict(os.environ, {ENV_NAME_PYLANCE_LOGGING_LEVEL: env_value}):
        assert get_log_level() == expected


def test_default_logger_level():
    assert LOGGER.level == logging.INFO


def test_set_logger_with_defaults(tmp_path):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file))
    assert LOGGER.level == logging.INFO
    assert len(LOGGER.handlers) == 1
    assert isinstance(LOGGER.handlers[0], logging.FileHandler)
    assert LOGGER.handlers[0].baseFilename == str(log_file)


def test_set_logger_with_custom_level(tmp_path):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file), level=logging.DEBUG)
    assert LOGGER.level == logging.DEBUG


def test_set_logger_with_custom_format(tmp_path):
    log_file = tmp_path / "test.log"
    custom_format = "%(levelname)s: %(message)s"
    set_logger(file_path=str(log_file), format_string=custom_format)
    print(LOGGER.handlers[0].formatter._fmt)
    assert LOGGER.handlers[0].formatter._fmt == custom_format


def test_set_logger_with_custom_handler(tmp_path):
    custom_handler = logging.StreamHandler()
    set_logger(log_handler=custom_handler)
    assert LOGGER.handlers[0] == custom_handler


def test_logger_output(tmp_path, caplog):
    log_file = tmp_path / "test.log"
    set_logger(file_path=str(log_file))
    with caplog.at_level(logging.INFO):
        LOGGER.info("Test log message")
    assert "Test log message" in caplog.text


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess does not work correctly in CI on Windows",
)
def test_timestamp_precision():
    def get_sample_log_line(precision: str):
        output = subprocess.run(
            [
                sys.executable,
                "-c",
                "import lance; import pyarrow as pa;\
                     lance.write_dataset(pa.table({'x': range(100)}), 'memory://test')",
            ],
            capture_output=True,
            check=True,
            env={
                "LANCE_LOG": "debug",
                "LANCE_LOG_TS_PRECISION": precision,
            },
        )
        return output.stderr.decode("utf-8").splitlines()[0]

    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}Z.*", get_sample_log_line("ns")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z.*", get_sample_log_line("us")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z.*", get_sample_log_line("ms")
    )
    assert re.match(
        r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z.*", get_sample_log_line("s")
    )
