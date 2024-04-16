# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import List

from lance import LanceDataset

def print_schema(dataset: LanceDataset) -> None:
    pass

def print_manifest(dataset: LanceDataset) -> None:
    pass

def list_transactions(
    dataset: LanceDataset,
    max_transactions: int = 10,
) -> List[str]:
    pass
