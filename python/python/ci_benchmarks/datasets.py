# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from functools import cache
from pathlib import Path

import lance
import requests
from lance.log import LOGGER


def is_on_google() -> bool:
    LOGGER.info("Testing if running on Google Cloud")
    try:
        rsp = requests.get("http://metadata.google.internal", timeout=5)
        LOGGER.info("Metadata-Flavor: %s", rsp.headers.get("Metadata-Flavor"))
        return rsp.headers["Metadata-Flavor"] == "Google"
    except requests.exceptions.RequestException as ex:
        LOGGER.info("Failed to connect to metadata server: %s", ex)
        return False


@cache
def _get_base_uri() -> str:
    if is_on_google():
        LOGGER.info("Running on Google Cloud, using gs://lance-benchmarks-ci-datasets/")
        return "gs://lance-benchmarks-ci-datasets/"
    else:
        data_path = Path.home() / "lance-benchmarks-ci-datasets"
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Running locally, using %s", data_path)
        return f"{data_path}/"


def get_dataset_uri(name: str) -> str:
    """Given a dataset name, return the URI appropriate for the current environment."""
    # This is a custom-built dataset, on a unique bucket, that is too big to reproduce
    # locally
    if name == "image_eda":
        if not is_on_google():
            raise ValueError("The image_eda dataset is only available on Google Cloud")
        return "gs://lance-benchmarks-ci-datasets/image_eda.lance"
    return f"{_get_base_uri()}{name}"


def open_dataset(name: str) -> lance.LanceDataset:
    if name.startswith("mem-"):
        if name == "mem-tpch":
            from ci_benchmarks.datagen.lineitems import gen_mem_tcph

            return gen_mem_tcph(data_storage_version="2.0")
        elif name == "mem-tpch-2.1":
            from ci_benchmarks.datagen.lineitems import gen_mem_tcph

            return gen_mem_tcph(data_storage_version="2.1")
        else:
            raise ValueError(f"Unknown memory dataset: {name}")
    else:
        return lance.dataset(get_dataset_uri(name))
