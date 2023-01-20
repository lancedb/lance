from .dataset import FileSystemDataset


def dataset(uri: str) -> FileSystemDataset:
    return FileSystemDataset(uri)

