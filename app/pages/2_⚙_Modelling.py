import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types 

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
        if target_feature_type in ["int", "float"]:
            task_type = "regression"
        else:
            task_type = "classification"
        
        st.write(f"Detected task type: {task_type}")

    
