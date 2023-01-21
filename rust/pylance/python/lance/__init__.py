from .dataset import LanceDataset, write_dataset


def dataset(uri: str) -> LanceDataset:
    return LanceDataset(uri)

