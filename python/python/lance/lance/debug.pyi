# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import List

from lance import LanceDataset
from lance.fragment import FragmentMetadata

def format_schema(dataset: LanceDataset) -> str:
    pass

def format_manifest(dataset: LanceDataset) -> str:
    pass

def format_fragment(fragment: FragmentMetadata, dataset: LanceDataset) -> str:
    pass

def list_transactions(
    dataset: LanceDataset,
    max_transactions: int = 10,
) -> List[str]:
    pass
