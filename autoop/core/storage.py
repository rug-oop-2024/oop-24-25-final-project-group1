from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the NotFoundError with the specified path.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for defining a storage interface.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            List[str]: List of paths.
        """
        pass


class LocalStorage(Storage):
    """
    LocalStorage implements the Storage interface for local file system.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage with a base path. Creates the base path
        directory if it does not exist.

        Args:
            base_path (str): The base path for local storage. Defaults to "./assets".
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a specific key under the base path.

        Args:
            data (bytes): Data to save.
            key (str): Key representing the relative path to save data.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a specific key under the base path.

        Args:
            key (str): Key representing the relative path to load data.

        Returns:
            bytes: Loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a specific key under the base path.

        Args:
            key (str): Key representing the relative path to delete data.
                Defaults to "/".
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all files under a specified prefix.

        Args:
            prefix (str): Prefix representing the relative path to list files.

        Returns:
            List[str]: List of file paths under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Assert that a given path exists. Raises NotFoundError if not.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with the given relative path.

        Args:
            path (str): Relative path to join.

        Returns:
            str: The full path.
        """
        return os.path.join(self._base_path, path)
