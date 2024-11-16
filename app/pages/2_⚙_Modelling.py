import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification.classification_models import get_classification_models
from autoop.core.ml.model.regression.regression_models import get_regression_models
from autoop.core.ml.metric import METRICS, get_metric

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

def initialize_automl():
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")
    return automl, datasets

def select_dataset(datasets):
    dataset_names = [dataset._name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    if selected_dataset_name:
        selected_dataset = next(dataset for dataset in datasets if dataset._name == selected_dataset_name)
        st.write(f"Selected dataset: {selected_dataset_name}")
        return selected_dataset
    return None

def select_features(features: list[Feature]):
    feature_names = [feature.name for feature in features]
    
    input_features_names = st.multiselect("Select input features", feature_names)

    input_features = [feature for feature in features if feature.name in input_features_names]

    available_target_features = [
        feature for feature in features if feature.name not in input_features_names
    ]

    target_feature_name = st.selectbox("Select target feature", [feature.name for feature in available_target_features])
    target_feature = next(feature for feature in available_target_features if feature.name == target_feature_name)

    return input_features, target_feature

def determine_task_type(features: list[Feature], target_feature: Feature) -> str:
    if target_feature.type == 'categorical':
        return 'classification'
    elif target_feature.type == 'numerical':
        return 'regression'
    else:
        raise ValueError(f"Unsupported target feature type: {target_feature.type}")

def select_model(task_type):
    if task_type == "classification":
        model_options = get_classification_models()
    else:
        model_options = get_regression_models()
    selected_model_name = st.selectbox("Select a model", list(model_options.keys()))
    selected_model_class = model_options[selected_model_name]
    st.write(f"Selected model: {selected_model_name}")
    return selected_model_class

def select_metrics():
    selected_metrics_names = st.multiselect("Select metrics", METRICS)
    selected_metrics = [get_metric(name) for name in selected_metrics_names]
    st.write(f"Selected metrics: {selected_metrics_names}")
    return selected_metrics, selected_metrics_names

def select_split_ratio():
    split_ratio = st.slider("Select the split ratio (0.1 to 0.9)", 0.1, 0.9, 0.8)
    st.write(f"Selected split ratio: {split_ratio}")
    return split_ratio

def create_pipeline(selected_dataset, input_features: list[Feature], target_feature: Feature, model, selected_metrics, split_ratio):
    raw = selected_dataset.read()
    
    if isinstance(raw, bytes):
        raw = pd.read_csv(io.StringIO(raw.decode()))
    
    dataset = Dataset.from_dataframe(raw, selected_dataset._name, selected_dataset._asset_path, selected_dataset._version)
    
    pipeline = Pipeline(
        dataset=dataset,
        input_features=input_features,
        target_feature=target_feature,
        model=model,
        metrics=selected_metrics,
        split=split_ratio
    )
    
    return pipeline

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl, datasets = initialize_automl()
selected_dataset = select_dataset(datasets)

if selected_dataset:
    features = detect_feature_types(selected_dataset)
    input_features, target_feature = select_features(features)

    if input_features and target_feature:
        task_type = determine_task_type(features, target_feature)
        st.write(f"Detected task type: {task_type}")

        model_class = select_model(task_type)
        model = model_class()

        print(f"Features: {features}")
        print(f"Input Features: {input_features}")
        print(f"Target Feature: {target_feature}")
        print(f"Dataset: {selected_dataset}")

        selected_metrics, selected_metrics_names = select_metrics()
        split_ratio = select_split_ratio()

        pipeline = create_pipeline(selected_dataset, input_features, target_feature, model, selected_metrics, split_ratio)

        st.write("## Pipeline Summary")
        st.write("### Dataset")
        st.write(f"Name: {selected_dataset._name}")
        st.write("### Features")
        st.write(f"Input Features: {', '.join([feature.name for feature in input_features])}")
        st.write(f"Target Feature: {target_feature.name}")
        st.write("### Model")
        st.write(f"Model: {model.__class__.__name__}")
        st.write("### Metrics")
        st.write(f"Metrics: {', '.join(selected_metrics_names)}")
        st.write("### Split Ratio")
        st.write(f"Split Ratio: {split_ratio}")
        st.write("Model metrics and predictions:", pipeline.execute())
