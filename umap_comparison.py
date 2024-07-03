import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from umap import umap_
from tqdm import tqdm
import plotly.express as px
import time
from sklearn.preprocessing import StandardScaler

# Define the features directory
features_dir = "/Users/neo/Documents/DATA/pooled_deepheme_data_features/test"


def get_all_class_names():
    """
    Get the names of all classes in the features directory.

    Returns:
    list: A list of strings, where each string is the name of a class.
    """
    class_names = [
        d
        for d in os.listdir(features_dir)
        if os.path.isdir(os.path.join(features_dir, d))
    ]
    return class_names


def load_features(class_names, n_samples_per_class):
    """
    Load and balance the features for the specified classes.

    Args:
    class_names (list of str): A list of class names.
    n_samples_per_class (int): Number of samples per class.

    Returns:
    combined_features (numpy array): Combined features from all classes.
    labels (list of str): Labels for each feature.
    """
    all_features = []
    labels = []

    for class_name in class_names:
        class_dir = os.path.join(features_dir, class_name)
        class_files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".npy")
        ]
        class_features = [
            np.load(file) for file in tqdm(class_files, desc=f"Loading {class_name}")
        ]
        class_features = np.vstack(class_features)

        # Bootstrap sampling to balance the number of samples per class
        if len(class_features) < n_samples_per_class:
            indices = np.random.choice(
                len(class_features), n_samples_per_class, replace=True
            )
        else:
            indices = np.random.choice(
                len(class_features), n_samples_per_class, replace=False
            )

        sampled_features = class_features[indices]

        all_features.append(sampled_features)
        labels.extend([class_name] * n_samples_per_class)

    combined_features = np.vstack(all_features)

    # Scale the data
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)

    return combined_features, labels


def plot_UMAP(umap_result, labels, class_names, n_components, dot_size):
    """
    Plot the UMAP components.

    Args:
    umap_result (numpy array): The UMAP result.
    labels (list of str): Labels for each point.
    class_names (list of str): The class names.
    n_components (int): Number of UMAP components.
    dot_size (int): Size of the dots in the plot.
    """
    if n_components == 2:
        fig = px.scatter(
            x=umap_result[:, 0],
            y=umap_result[:, 1],
            color=labels,
            size=np.full(len(labels), dot_size),  # Setting the size of the dots
            labels={"x": "UMAP Component 1", "y": "UMAP Component 2"},
            title="UMAP Components",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif n_components == 3:
        fig = px.scatter_3d(
            x=umap_result[:, 0],
            y=umap_result[:, 1],
            z=umap_result[:, 2],
            color=labels,
            size=np.full(len(labels), dot_size),  # Setting the size of the dots
            labels={
                "x": "UMAP Component 1",
                "y": "UMAP Component 2",
                "z": "UMAP Component 3",
            },
            title="UMAP Components",
        )
        st.plotly_chart(fig, use_container_width=True)


st.title("UMAP Visualization of Features")
class_names = get_all_class_names()

# UI for selecting class names
selected_class_names = st.multiselect(
    "Select Class Names", class_names, default=class_names
)
if st.checkbox("Select All", value=True):
    selected_class_names = class_names

# UI for selecting UMAP parameters
n_components = st.radio("Number of Components", [2, 3], index=0)
n_samples_per_class = st.slider(
    "Number of Samples per Class", min_value=10, max_value=1000, value=100
)
k = n_samples_per_class

# Suggest a good starting point for n_neighbors
suggested_n_neighbors = int(np.sqrt(k))

n_neighbors = st.slider(
    "Number of Neighbors", min_value=5, max_value=50, value=suggested_n_neighbors
)
min_dist = st.slider("Minimum Distance", min_value=0.0, max_value=1.0, value=0.1)
n_epochs = st.slider("Number of Epochs", min_value=100, max_value=5000, value=500)
dot_size = st.slider(
    "Dot Size", min_value=1, max_value=20, value=2
)  # Default dot size is set to 2

if st.button("Submit"):
    combined_features, labels = load_features(selected_class_names, n_samples_per_class)

    st.write(f"Shape of combined features: {combined_features.shape}")
    st.write(f"Number of labels: {len(labels)}")
    st.write(f"Unique labels: {np.unique(labels)}")

    start_time = time.time()

    with st.spinner("Running UMAP..."):
        # reducer = umap.UMAP(
        #     n_components=n_components,
        #     n_neighbors=n_neighbors,
        #     min_dist=min_dist,
        #     n_epochs=n_epochs,
        #     metric='euclidean'
        # )

        # Apply UMAP to the extracted features
        reducer = umap_.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
        )

        umap_result = reducer.fit_transform(combined_features)

    end_time = time.time()
    elapsed_time = end_time - start_time

    st.success(f"UMAP fitting completed in {elapsed_time:.2f} seconds.")
    st.write(f"Shape of UMAP result: {umap_result.shape}")

    plot_UMAP(umap_result, labels, selected_class_names, n_components, dot_size)
