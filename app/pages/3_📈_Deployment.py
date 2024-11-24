import streamlit as st
import os
import pickle
import pandas as pd


st.set_page_config(page_title="Deployment", page_icon="📈")


def load_pipelines(directory: str) -> list[str]:
    """
    Loads the list of saved pipelines from the specified directory

    Args:
        directory (str): The directory to load pipelines from

    Returns:
        list[str]: A list of pipeline file paths
    """
    pipelines = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            pipelines.append(os.path.join(directory, filename))
    return pipelines


def load_pipeline(file_path: str) -> dict:
    """
    Loads a pipeline from the specified file path

    Args:
        file_path (str): The file path to load the pipeline from

    Returns:
        dict: The loaded pipeline data
    """
    with open(file_path, "rb") as file:
        pipeline_data = pickle.load(file)
    return pipeline_data


def show_pipeline_summary(pipeline_data: dict) -> None:
    """
    Shows the summary of the selected pipeline

    Args:
        pipeline_data (dict): The pipeline data to display

    Returns:
        None
    """
    st.subheader("Pipeline Summary")
    st.markdown("### Name")
    st.write(pipeline_data["name"])
    st.markdown("### Version")
    st.write(pipeline_data["version"])
    st.markdown("### Input Features")
    st.write(pipeline_data["input_features"])
    st.markdown("### Target Feature")
    st.write(pipeline_data["target_feature"])
    st.markdown("### Data Split")
    st.write(pipeline_data["split"])
    st.markdown("### Model")
    st.write(pipeline_data["model"])
    st.markdown("### Metrics")
    st.write(pipeline_data["metrics"])


def predict_with_pipeline(pipeline_data: dict, prediction_data: pd.DataFrame
                          ) -> pd.DataFrame:
    """
    Performs predictions using the loaded pipeline.

    Args:
        pipeline_data (dict): The loaded pipeline data.
        prediction_data (pd.DataFrame): The input data for predictions.

    Returns:
        pd.DataFrame: A DataFrame containing the original data and predictions.
    """
    model = pipeline_data["model"]
    predictions = model.predict(prediction_data)

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.iloc[:, 0]
    elif len(predictions.shape) == 2:
        predictions = predictions[:, 0]

    results = pd.DataFrame(prediction_data)
    results["Prediction"] = predictions

    return results


st.title("📈 Deployment")

pipelines_directory = "assets/objects/pipelines"
pipelines = load_pipelines(pipelines_directory)

if pipelines:
    selected_pipeline_path = st.selectbox("Select a pipeline", pipelines)
    if selected_pipeline_path:
        pipeline_data = load_pipeline(selected_pipeline_path)
        show_pipeline_summary(pipeline_data)

        st.subheader("Upload Data for Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data")
            st.dataframe(input_data)

            input_features_names = [
                feature.name for feature in pipeline_data["input_features"]
            ]
            target_feature_name = pipeline_data["target_feature"].name

            st.write("### Validation Check")
            missing_input_features = [
                feature for feature in input_features_names
                if feature not in input_data.columns
            ]
            target_present = target_feature_name in input_data.columns

            if not missing_input_features:
                prediction_data = input_data[input_features_names]
                target_data = (
                    input_data[target_feature_name] if target_present else None
                )

                if target_present:
                    st.success(
                        "The uploaded file contains all required features, "
                        "including the target feature."
                    )
                else:
                    st.warning(
                        "The target feature is missing. Predictions will "
                        "still proceed."
                    )

                predictions = predict_with_pipeline(
                    pipeline_data, prediction_data
                    )
                st.write("### Prediction Results")
                st.dataframe(predictions)

                csv_data = predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(
                    "The uploaded file is missing the "
                    "following input features: "
                    f"{missing_input_features}"
                )
else:
    st.write("No saved pipelines found.")
