class Feature:
    def __init__(self, name: str, type: str):
        """
        Represents a feature in a dataset.

        :param name: The name of the feature (e.g., column name in the DataFrame).
        :param type: The type of the feature (e.g., "numerical", "categorical").
        """
        self.name = name
        self.type = type

    def __repr__(self):
        return f"Feature(name={self.name}, type={self.type})"

    def is_numerical(self) -> bool:
        """
        Checks if the feature is numerical.

        :return: True if the feature is numerical, otherwise False.
        """
        return self.type == "numerical"

    def is_categorical(self) -> bool:
        """
        Checks if the feature is categorical.

        :return: True if the feature is categorical, otherwise False.
        """
        return self.type == "categorical"
    