import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types 
from autoop.core.ml.model.classification.classification_models import get_classification_models
from autoop.core.ml.model.regression.regression_models import get_regression_models


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

dataset_names = [dataset._name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

if selected_dataset_name:
    selected_dataset = next(dataset for dataset in datasets if dataset._name == selected_dataset_name)
    st.write(f"Selected dataset: {selected_dataset_name}")
    
    features = detect_feature_types(selected_dataset)
    feature_names = [feature.name for feature in features]
    input_features = st.multiselect("Select input features", feature_names)
    available_target_features = [feature for feature in feature_names if feature not in input_features]
    
    target_features = st.selectbox("Select target feature", available_target_features)

    if input_features and target_features:
        target_feature_type = next(feature for feature in features if feature.name == target_features).type
        if target_feature_type in ["numerical"]:
            task_type = "regression"
        else:
            task_type = "classification"
        
        st.write(f"Detected task type: {task_type}")
        
        if task_type == "classification":
            model_options = get_classification_models()
        else:
            model_options = get_regression_models()

        selected_model_name = st.selectbox("Select a model", list(model_options.keys()))
        selected_model_class = model_options[selected_model_name]
        st.write(f"Selected model: {selected_model_name}")
        
        split_value = st.slider("Select a dataset split (e.g., 0.8 for 80% training, 20% testing)", 0.1, 0.9, 0.8)
        st.write(f"Selected split value: {split_value}")

        pipeline = Pipeline(
            metrics=[],
            dataset=selected_dataset,
            model=selected_model_class(),
            input_features=[feature for feature in features if feature.name in input_features],
            target_feature=next(feature for feature in features if feature.name == target_features),
            split=split_value
        )

        st.write("Pipeline created successfully!")
        st.write(pipeline)


