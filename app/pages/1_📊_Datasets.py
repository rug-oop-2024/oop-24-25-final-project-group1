import os
import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()


def handle_file_upload() -> None:
    """
    Handles the file upload process for CSV files. Users can upload a CSV file,
    view its contents, and save it as a dataset with a specified name and 
    version.
    """
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.write(data)

            dataset_name = st.text_input("Enter dataset name:", value="MyDataset")
            version = st.text_input("Enter version:", value="1.0.0")

            asset_base_dir = "datasets"
            os.makedirs(asset_base_dir, exist_ok=True)
            asset_path = f"{asset_base_dir}/{dataset_name}_v{version}.csv"

            if st.button("Save Dataset"):
                if dataset_name and asset_path and version:
                    dataset = Dataset.from_dataframe(
                        data=data,
                        name=dataset_name,
                        asset_path=asset_path,
                        version=version
                    )
                    automl.registry.register(dataset)
                    st.success(
                        f"Dataset '{dataset_name}' saved successfully!"
                    )
                else:
                    st.error(
                        "Please enter all required fields: dataset name, "
                        "asset path, and version."
                    )
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV file: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


def display_existing_datasets() -> None:
    """
    Displays a list of datasets currently registered in the AutoML system,
    including their names and versions.
    """
    st.subheader("Existing Datasets")
    datasets = automl.registry.list(type="dataset")
    for ds in datasets:
        st.write(f"Dataset Name: {ds._name}, Version: {ds._version}")


st.title("Dataset Management")
handle_file_upload()
display_existing_datasets()
