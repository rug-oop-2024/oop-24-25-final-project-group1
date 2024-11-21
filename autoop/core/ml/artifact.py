import base64
import itertools
import os

from abc import ABC


class Artifact(ABC):
    """Abstract base class representing an asset that can be stored and 
    contains information about this specific asset."""

    _id_iter = itertools.count()
    _base_path = "assets/objects/"

    def __init__(
        self,
        name: str,
        asset_path: str,
        version: str,
        data: bytes = None,
        metadata: dict = None,
        type: str = "",
        tags: list = None
    ) -> None:
        """
        Initializes an Artifact instance.

        Args:
            name (str): The name of the asset.
            asset_path (str): The path where the asset is stored.
            version (str): The version of the asset.
            data (bytes, optional): The binary data of the asset. Defaults to None.
            metadata (dict, optional): A dictionary containing additional metadata
                about the asset. Defaults to an empty dictionary if None.
            type (str, optional): The type of the asset, such as "model:torch"
                or "dataset". Defaults to an empty string.
            tags (list, optional): A list of tags describing the asset.
                Defaults to an empty list if None.
        """
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata if metadata is not None else {}
        self._type = type
        self._tags = tags if tags is not None else []
        self._id = str(next(Artifact._id_iter))

    @property
    def name(self) -> str:
        return self._name

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @property
    def version(self) -> str:
        return self._version

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def type(self) -> str:
        return self._type

    @property
    def tags(self) -> list:
        return self._tags

    @property
    def id(self) -> str:
        return self._id

    def save1(self, directory: str) -> None:
        """
        Saves the artifact data to a specified directory. If the directory does
        not exist, it creates it.

        Args:
            directory (str): The directory where the data will be saved.
        """
        print(f"The dir path is path is {directory}")

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory + "/" + self._asset_path}", 'wb') as file:
            file.write(self._data)

    def read(self) -> bytes:
        """
        Reads and returns the binary data of the artifact. If the artifact file
        does not exist, it creates a new file in the "exports" directory.

        Returns:
            bytes: The binary data of the artifact.

        Raises:
            FileNotFoundError: If the file at `asset_path` could not be found
                after attempting to read it.
        """
        if not os.path.exists(self._base_path + self._asset_path):
            print(f"The path is {self._base_path + self._asset_path}")
            # Providing a warning instead of raising an error for testing purposes
            print("Warning: Creating the data because it does not exist.")
            print(f"The path is {self._asset_path}")
            self.save1(self._base_path)
        try:
            with open(self._base_path + self._asset_path, 'rb') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file at path {self._base_path + self._asset_path} could not be found."
            )

    def get_id(self) -> str:
        """
        Generates a unique ID for the artifact based on its asset path and
        version.

        Returns:
            str: A unique ID string derived from base64 encoding of the asset
                path and version.
        """
        encoded_path = base64.urlsafe_b64encode(self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"

    def get_metadata(self) -> dict:
        """
        Retrieves the metadata of the artifact.

        Returns:
            dict: A dictionary containing the metadata.
        """
        return self._metadata

    def update_metadata(self, key: str, value) -> None:
        """
        Updates the metadata of the artifact with a new key-value pair.

        Args:
            key (str): The key to update or add.
            value: The value associated with the key.
        """
        self._metadata[key] = value
