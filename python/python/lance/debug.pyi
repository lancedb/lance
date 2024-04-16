# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import List

from lance import LanceDataset
from lance.fragment import FragmentMetadata

def print_schema(dataset: LanceDataset) -> None:
    pass

def print_manifest(dataset: LanceDataset) -> None:
    pass

def print_fragment(fragment: FragmentMetadata) -> None:
    pass

def list_transactions(
    dataset: LanceDataset,
    max_transactions: int = 10,
) -> List[str]:
    pass
