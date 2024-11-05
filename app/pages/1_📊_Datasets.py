from app.core.system import AutoMLSystem
import os
import streamlit as st
import pandas as pd
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


st.title("Dataset Management")

uploaded_file = st.file_uploader("Upload a CSV file to create a dataset", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    dataset_name = st.text_input("Enter a name for the dataset:", value="MyDataset")
    version = st.text_input("Enter version:", value="1.0.0")

    asset_base_dir = "datasets"
    
    os.makedirs(asset_base_dir, exist_ok=True)

    asset_path = f"{asset_base_dir}/{dataset_name}_v{version}.csv"
    
    dataset = Dataset.from_dataframe(data=data, name=dataset_name, asset_path=asset_path, version=version)

    st.write("Dataset Preview:", data.head())

    if st.button("Save Dataset"):
        automl_system = AutoMLSystem.get_instance()
        automl_system.registry.register(dataset)
        st.success(f"Dataset '{dataset_name}' has been saved successfully!")

st.subheader("Existing Datasets")
automl_system = AutoMLSystem.get_instance()
datasets = automl_system.registry.list(type="dataset")
for ds in datasets:
    st.write(f"Dataset Name: {ds._name}, Version: {ds._version}")




