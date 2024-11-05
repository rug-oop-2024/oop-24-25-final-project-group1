from app.core.system import AutoMLSystem
import os
import streamlit as st
import pandas as pd
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


st.title("Dataset Management")

# Step 1: Upload a CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file to create a dataset", type="csv")
if uploaded_file:
    # Read the uploaded CSV into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Get dataset name and version from the user
    dataset_name = st.text_input("Enter a name for the dataset:", value="MyDataset")
    version = st.text_input("Enter version:", value="1.0.0")

    # Define base directory for dataset assets within "assets/objects"
    asset_base_dir = "datasets"
    
    # Ensure the base directory exists
    os.makedirs(asset_base_dir, exist_ok=True)

    # Create a unique asset path for the dataset
    asset_path = f"{asset_base_dir}/{dataset_name}_v{version}.csv"
    
    # Create the Dataset artifact using the factory method
    dataset = Dataset.from_dataframe(data, name=dataset_name, asset_path=asset_path, version=version)

    # Show preview of the data
    st.write("Dataset Preview:", data.head())

    # Step 3: Save the dataset in the artifact registry
    if st.button("Save Dataset"):
        # Use the AutoMLSystem singleton to access the registry
        automl_system = AutoMLSystem.get_instance()
        automl_system.registry.register(dataset)
        st.success(f"Dataset '{dataset_name}' has been saved successfully!")

# Step 4: List existing datasets in the registry
st.subheader("Existing Datasets")
automl_system = AutoMLSystem.get_instance()
datasets = automl_system.registry.list(type="dataset")
for ds in datasets:
    st.write(f"Dataset Name: {ds.name}, Version: {ds.version}")




