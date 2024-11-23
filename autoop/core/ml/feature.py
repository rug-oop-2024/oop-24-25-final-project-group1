class Feature:
    """
    Represents a feature in a dataset.

    Attributes:
        name (str): The name of the feature
        type (str): The type of the feature
    """

    def __init__(self, name: str, type: str) -> None:
        """
        Initializes a Feature instance.

        Args:
            name (str): The name of the feature
            type (str): The type of the feature
        """
        self.name = name
        self.type = type

    def __repr__(self) -> str:
        """
        Returns a string representation of the Feature instance.

        Returns:
            str: A string describing the Feature instance.
        """
        return f"Feature(name={self.name}, type={self.type})"

    def is_numerical(self) -> bool:
        """
        Checks if the feature is numerical.

        Returns:
            bool: True if the feature is numerical, otherwise False.
        """
        return self.type == "numerical"

    def is_categorical(self) -> bool:
        """
        Checks if the feature is categorical.

        Returns:
            bool: True if the feature is categorical, otherwise False.
        """
        return self.type == "categorical"
