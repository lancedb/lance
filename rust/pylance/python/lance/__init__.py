from .dataset import LanceDataset


def dataset(uri: str) -> LanceDataset:
    return LanceDataset(uri)

