from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List, Optional


class ArtifactRegistry:
    """
    Manages the registration, retrieval, and deletion of artifacts in the
    AutoML system.

    Methods:
        register: Registers a new artifact in the system.
        list: Lists all artifacts, optionally filtered by type.
        get: Retrieves a specific artifact by its ID.
        delete: Deletes an artifact from the system.
    """

    def __init__(
        self, database: Database, storage: Storage
    ) -> None:
        """
        Initializes the ArtifactRegistry.

        Args:
            database (Database): The database instance for metadata storage.
            storage (Storage): The storage instance for artifact data storage.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers a new artifact by saving its data and metadata.

        Args:
            artifact (Artifact): The artifact to register.

        Returns:
            None
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: Optional[str] = None) -> List[Artifact]:
        """
        Lists all artifacts, optionally filtered by type.

        Args:
            type (Optional[str]): The type of artifact to filter by.

        Returns:
            List[Artifact]: A list of matching artifacts.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The unique ID of the artifact.

        Returns:
            Artifact: The retrieved artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID.

        Args:
            artifact_id (str): The unique ID of the artifact to delete.

        Returns:
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class representing the AutoML system.

    Manages the artifact registry, storage, and database for machine learning
    operations.

    Attributes:
        _storage (LocalStorage): The local storage instance.
        _database (Database): The database instance.
        _registry (ArtifactRegistry): The artifact registry instance.
    """

    _instance: Optional["AutoMLSystem"] = None

    def __init__(
        self, storage: LocalStorage, database: Database
    ) -> None:
        """
        Initializes the AutoMLSystem.

        Args:
            storage (LocalStorage): The local storage instance.
            database (Database): The database instance.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieves the singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Accesses the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
