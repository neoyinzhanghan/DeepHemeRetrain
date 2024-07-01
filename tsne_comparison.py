import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import time

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
        class_features = [np.load(file) for file in tqdm(class_files, desc=f"Loading {class_name}")]
        class_features = np.vstack(class_features)
        
        # Bootstrap sampling to balance the number of samples per class
        if len(class_features) < n_samples_per_class:
            indices = np.random.choice(len(class_features), n_samples_per_class, replace=True)
        else:
            indices = np.random.choice(len(class_features), n_samples_per_class, replace=False)
        
        sampled_features = class_features[indices]
        
        all_features.append(sampled_features)
        labels.extend([class_name] * n_samples_per_class)

    combined_features = np.vstack(all_features)
    return combined_features, labels

def plot_TSNE(tsne_result, labels, class_names, n_components, dot_size):
    """
    Plot the t-SNE components.

    Args:
    tsne_result (numpy array): The t-SNE result.
    labels (list of str): Labels for each point.
    class_names (list of str): The class names.
    n_components (int): Number of t-SNE components.
    dot_size (int): Size of the dots in the plot.
    """
    if n_components == 2:
        fig = px.scatter(
            x=tsne_result[:, 0], 
            y=tsne_result[:, 1], 
            color=labels, 
            size=np.full(len(labels), dot_size),  # Setting the size of the dots
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
            title="t-SNE Components"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif n_components == 3:
        fig = px.scatter_3d(
            x=tsne_result[:, 0], 
            y=tsne_result[:, 1], 
            z=tsne_result[:, 2], 
            color=labels,
            size=np.full(len(labels), dot_size),  # Setting the size of the dots
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'z': 't-SNE Component 3'},
            title="t-SNE Components"
        )
        st.plotly_chart(fig, use_container_width=True)

st.title("t-SNE Visualization of Features")
class_names = get_all_class_names()

# UI for selecting class names
selected_class_names = st.multiselect("Select Class Names", class_names, default=class_names)
if st.checkbox("Select All", value=True):
    selected_class_names = class_names

# UI for selecting t-SNE parameters
n_components = st.radio("Number of Components", [2, 3], index=0)
perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
n_iter = st.slider("Number of Iterations", min_value=250, max_value=1000, value=250)
n_samples_per_class = st.slider("Number of Samples per Class", min_value=10, max_value=1000, value=100)
dot_size = st.slider("Dot Size", min_value=1, max_value=20, value=2)  # Default dot size is set to 2

if st.button("Submit"):
    combined_features, labels = load_features(selected_class_names, n_samples_per_class)

    st.write(f"Combined features shape: {combined_features.shape}")
    st.write(f"Number of labels: {len(labels)}")
    
    start_time = time.time()
    
    with st.spinner("Running t-SNE..."):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, verbose=1)
        tsne_result = tsne.fit_transform(combined_features)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    st.success(f"t-SNE fitting completed in {elapsed_time:.2f} seconds.")
    st.write(f"t-SNE result shape: {tsne_result.shape}")
    
    plot_TSNE(tsne_result, labels, selected_class_names, n_components, dot_size)
