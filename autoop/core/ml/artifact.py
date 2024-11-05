import base64
from abc import ABC, abstractmethod
import itertools
import os

class Artifact(ABC):
    id_iter = itertools.count()
    def __init__(self, name: str, asset_path: str, version: str, data: bytes = None, metadata: dict = None, type: str = "", tags: list = None) -> None:
        """
        Abstract base class representing an asset that can be stored and contains information about this specific asset.

        :param name: The name of the asset.
        :param asset_path: The path where the asset is stored.
        :param version: The version of the asset.
        :param data: The binary data of the asset (optional).
        :param metadata: A dictionary containing additional metadata about the asset (optional).
        :param type: The type of the asset, such as "model:torch" or "dataset" (optional).
        :param tags: A list of tags describing the asset (optional).
        """
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.metadata = metadata if metadata is not None else {}
        self.type = type
        self.tags = tags if tags is not None else []
        self.id = str(next(Artifact.id_iter))
        
    def save1(self, directory: str) -> None:
        """
        Saves the data into the artifact.
        """
        print(f"The dir path is path is {directory}")

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)

    def read(self) -> bytes:
        """
        Reads the data from the artifact.
        """
        if not os.path.exists(self.asset_path):
            print(f"The path is {self.asset_path}")
            # Providing a warning instead of raising an error for testing purposes
            print(f"Warning: Creating the data because it does not exist you loser")
            self.save1("exports")
            # return b""
        try:
            with open(self.asset_path, 'rb') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path {self.asset_path} could not be found.")

    def get_id(self) -> str:
        """
        Generate the unique ID of the artifact based on asset path and version.

        :return: A unique ID string derived from base64 encoding of the asset path and the version.
        """
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def get_metadata(self) -> dict:
        """
        Retrieve the metadata of the artifact.

        :return: A dictionary containing the metadata.
        """
        return self.metadata

    def update_metadata(self, key: str, value) -> None:
        """
        Update the metadata of the artifact with a new key-value pair.

        :param key: The key to update or add.
        :param value: The value associated with the key.
        """
        self.metadata[key] = value
