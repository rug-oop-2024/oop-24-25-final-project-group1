import streamlit as st
import os
import pickle
import pandas as pd

st.set_page_config(page_title="Deployment", page_icon="📈")

def load_pipelines(directory: str) -> list:
    """
    Loads the list of saved pipelines from the specified directory.

    Args:
        directory (str): The directory to load pipelines from.

    Returns:
        list: A list of pipeline file paths.
    """
    pipelines = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            pipelines.append(os.path.join(directory, filename))
    return pipelines

def load_pipeline(file_path: str):
    """
    Loads a pipeline from the specified file path.

    Args:
        file_path (str): The file path to load the pipeline from.

    Returns:
        dict: The loaded pipeline data.
    """
    with open(file_path, 'rb') as file:
        pipeline_data = pickle.load(file)
    #append the file path to the pipeline data
    filename = os.path.basename(file_path)
    name, version = os.path.splitext(filename)[0].rsplit('_v', 1)
    pipeline_data['name'] = name
    pipeline_data['version'] = version
    return pipeline_data

def show_pipeline_summary(pipeline_data: dict) -> None:
    """
    Displays the summary of the selected pipeline.

    Args:
        pipeline_data (dict): The pipeline data to display.
    """
    st.subheader("Pipeline Summary")
    st.markdown("### Name")
    st.write(pipeline_data['name'])
    st.markdown("### Version")
    st.write(pipeline_data['version'])

st.title("📈 Deployment")

pipelines_directory = "assets/objects/pipelines"
pipelines = load_pipelines(pipelines_directory)

if pipelines:
    selected_pipeline_path = st.selectbox("Select a pipeline", pipelines)
    if selected_pipeline_path:
        pipeline_data = load_pipeline(selected_pipeline_path)
        show_pipeline_summary(pipeline_data)
else:
    st.write("No saved pipelines found.")


