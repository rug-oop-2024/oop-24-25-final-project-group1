import streamlit as st
import pandas as pd
import io
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification.classification_models import (
    get_classification_models,
)
from autoop.core.ml.model.regression.regression_models import (
    get_regression_models,
)
from autoop.core.ml.metric import (
    get_metric,
    get_classification_metrics,
    get_regression_metrics,
)

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Writes helper text in a styled markdown format.

    Args:
        text (str): The helper text to display.
    """
    st.markdown(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


def initialize_automl() -> tuple[AutoMLSystem, list[Dataset]]:
    """
    Initializes the AutoML system and retrieves the list of datasets.

    Returns:
        tuple[AutoMLSystem, list[Dataset]]: The AutoML system instance and
        the list of datasets.
    """
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")
    return automl, datasets


def select_dataset(datasets: list[Dataset]) -> Dataset:
    """
    Allows the user to select a dataset from the list of datasets.

    Args:
        datasets (list[Dataset]): The list of available datasets.

    Returns:
        Dataset: The selected dataset.
    """
    dataset_names = [dataset._name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    if selected_dataset_name:
        selected_dataset = next(
            dataset for dataset in datasets
            if dataset._name == selected_dataset_name
        )
        st.write(f"Selected dataset: {selected_dataset_name}")
        return selected_dataset
    return None


def select_features(features: list[Feature]) -> tuple[list[Feature], Feature]:
    """
    Allows the user to select input features and a target feature from the
    list of features.

    Args:
        features (list[Feature]): The list of available features.

    Returns:
        tuple[list[Feature], Feature]: The selected input features and the
        target feature.
    """
    feature_names = [feature.name for feature in features]

    input_features_names = st.multiselect(
        "Select input features", feature_names
    )

    input_features = [
        feature for feature in features if feature.name in
        input_features_names
    ]

    available_target_features = [
        feature for feature in features if feature.name not in
        input_features_names
    ]

    target_feature_name = st.selectbox(
        "Select target feature",
        [feature.name for feature in available_target_features],
        key="target_feature",
    )
    target_feature = next(
        feature for feature in available_target_features
        if feature.name == target_feature_name
    )

    return input_features, target_feature


def determine_task_type(
    features: list[Feature], target_feature: Feature
) -> str:
    """
    Determines the task type (classification or regression) based on the
    target feature type.

    Args:
        features (list[Feature]): The list of features.
        target_feature (Feature): The target feature.

    Returns:
        str: The task type ("classification" or "regression").
    """
    if target_feature.type == "categorical":
        return "classification"
    elif target_feature.type == "numerical":
        return "regression"
    else:
        raise ValueError(
            f"Unsupported target feature type: {target_feature.type}"
        )


def select_model(task_type: str) -> type:
    """
    Allows the user to select a model based on the task type.

    Args:
        task_type (str): The task type ("classification" or "regression").

    Returns:
        type: The selected model class.
    """
    if task_type == "classification":
        model_options = get_classification_models()
    else:
        model_options = get_regression_models()
    selected_model_name = st.selectbox(
        "Select a model", list(model_options.keys())
    )
    selected_model_class = model_options[selected_model_name]
    st.write(f"Selected model: {selected_model_name}")
    return selected_model_class


def select_metrics(task_type: str) -> tuple[list, list[str]]:
    """
    Allows the user to select metrics based on the task type.

    Args:
        task_type (str): The task type ("classification" or "regression").

    Returns:
        tuple[list, list[str]]: The selected metrics and their names.
    """
    if task_type == "classification":
        available_metrics = get_classification_metrics()
    else:
        available_metrics = get_regression_metrics()

    selected_metrics_names = st.multiselect(
        "Select metrics",
        [metric.__str__() for metric in available_metrics]
    )
    selected_metrics = [get_metric(name) for name in selected_metrics_names]
    st.write(
        "Selected metrics: "
        f"{[metric.__str__() for metric in selected_metrics]}"
    )
    return selected_metrics, selected_metrics_names


def select_split_ratio() -> float:
    """
    Allows the user to select the split ratio for the dataset.

    Returns:
        float: The selected split ratio.
    """
    split_ratio = st.slider(
        "Select the split ratio (0.1 to 0.9)", 0.1, 0.9, 0.8
    )
    st.write(f"Selected split ratio: {split_ratio}")
    return split_ratio


def create_pipeline(
    selected_dataset: Dataset,
    input_features: list[Feature],
    target_feature: Feature,
    model: type,
    selected_metrics: list,
    split_ratio: float,
) -> Pipeline:
    """
    Creates a pipeline with the selected parameters.

    Args:
        selected_dataset (Dataset): The selected dataset.
        input_features (list[Feature]): The selected input features.
        target_feature (Feature): The selected target feature.
        model (type): The selected model.
        selected_metrics (list): The selected metrics.
        split_ratio (float): The selected split ratio.

    Returns:
        Pipeline: The created pipeline.
    """
    raw = selected_dataset.read()

    if isinstance(raw, bytes):
        raw = pd.read_csv(io.StringIO(raw.decode()))

    dataset = Dataset.from_dataframe(
        raw, selected_dataset._name, selected_dataset._asset_path,
        selected_dataset._version
    )

    pipeline = Pipeline(
        dataset=dataset,
        input_features=input_features,
        target_feature=target_feature,
        model=model,
        metrics=selected_metrics,
        split=split_ratio,
    )

    return pipeline


def to_artifact(pipeline: Pipeline, name: str, version: str) -> Artifact:
    """
    Converts a given pipeline into an artifact.

    Args:
        pipeline (Pipeline): The pipeline to convert.
        name (str): The name of the artifact.
        version (str): The version of the artifact.

    Returns:
        Artifact: The created artifact from the pipeline.
    """
    pipeline_data = pickle.dumps({
        "name": name,
        "version": version,
        "input_features": pipeline._input_features,
        "target_feature": pipeline._target_feature,
        "split": pipeline._split,
        "model": pipeline._model,
        "metrics": pipeline._metrics,
        "dataset": pipeline._dataset,
    })

    artifact = Artifact(
        name=name,
        asset_path=f"{name}_v{version}.pkl",
        version=version,
        data=pipeline_data,
        metadata={
            "input_features": [
                feature.name for feature in pipeline._input_features
            ],
            "target_feature": pipeline._target_feature.name,
            "split": pipeline._split,
            "model_type": pipeline._model.type,
            "metrics": [
                metric.__str__() for metric in pipeline._metrics
            ],
            "dataset_name": pipeline._dataset._name
        },
        type="pipeline",
        tags=["pipeline", "ml", "automl"]
    )

    return artifact


Pipeline.to_artifact = to_artifact


def save_pipeline(pipeline: Pipeline) -> None:
    """
    Prompts the user to give a name and version for the pipeline and saves it.

    Args:
        pipeline (Pipeline): The pipeline to save.
    """
    st.subheader("Save Pipeline")
    pipeline_name = st.text_input("Enter pipeline name")
    pipeline_version = st.text_input("Enter pipeline version")

    if st.button("Save Pipeline"):
        if pipeline_name and pipeline_version:
            artifact = pipeline.to_artifact(
                name=pipeline_name, version=pipeline_version
            )
            artifact.save1("assets/objects/pipelines")
            st.success(
                f"Pipeline '{pipeline_name}' (version {pipeline_version}) "
                "saved successfully!"
            )
        else:
            st.error(
                "Please provide both a name and a version for the pipeline."
            )


st.title("⚙ Modelling")
write_helper_text(
    "This is a Machine Learning pipeline that trains a model on a dataset."
)

automl, datasets = initialize_automl()
selected_dataset = select_dataset(datasets)

if selected_dataset:
    features = detect_feature_types(selected_dataset)
    input_features, target_feature = select_features(features)

    if input_features and target_feature:
        task_type = determine_task_type(features, target_feature)
        st.subheader(f"Detected task type: {task_type}")

        model_class = select_model(task_type)
        model = model_class()

        selected_metrics, selected_metrics_names = (
            select_metrics(task_type)
        )
        split_ratio = select_split_ratio()

        pipeline = create_pipeline(
            selected_dataset,
            input_features,
            target_feature,
            model,
            selected_metrics,
            split_ratio,
        )
        # display a summary of the pipeline
        st.subheader("Pipeline Summary")
        st.markdown("### Dataset")
        st.write(f"**Name:** {selected_dataset._name}")
        st.markdown("### Features")
        st.write(
            "**Input Features:** "
            f"{', '.join([feature.name for feature in input_features])}"
        )
        st.write(f"**Target Feature:** {target_feature.name}")
        st.markdown("### Model")
        st.write(f"**Model:** {model.__class__.__name__}")
        st.markdown("### Metrics")
        st.write(f"**Metrics:** {', '.join(selected_metrics_names)}")
        st.markdown("### Split Ratio")
        st.write(f"**Split Ratio:** {split_ratio}")
        st.subheader("Model metrics and predictions")
        st.write(pipeline.execute())

        save_pipeline(pipeline)
