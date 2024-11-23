from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class representing a dataset, inheriting from the Artifact base class.
    This class specifically handles datasets and provides additional
    functionality for reading and saving data in pandas DataFrame format.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Dataset instance, setting its type to "dataset".

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, 
        name: str, 
        asset_path: str, 
        version: str = "1.0.0"
    ) -> 'Dataset':
        """
        Creates a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be saved as a dataset.
            name (str): The name of the dataset.
            asset_path (str): The path where the dataset will be stored.
            version (str, optional): The version of the dataset. Defaults to
                "1.0.0".

        Returns:
            Dataset: A Dataset instance containing the serialized DataFrame.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the given pandas DataFrame as a dataset.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The serialized DataFrame in bytes format.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save1(bytes)
