import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

features_dir = "/Users/neo/Documents/DATA/pooled_deepheme_data_features/train"

def get_all_class_names():
    class_names = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
    return class_names

def load_features(class_names):
    all_features = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(features_dir, class_name)
        class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".npy")]
        class_features = [np.load(file) for file in class_files]
        class_features = np.vstack(class_features)
        all_features.append(class_features)
        labels.extend([class_name] * len(class_features))
    combined_features = np.vstack(all_features)
    return combined_features, labels

def plot_pca_2d(class_names, dot_size):
    combined_features, labels = load_features(class_names)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_features)

    plt.style.use('dark_background')
    plt.figure(figsize=(16, 12))
    for class_name in class_names:
        indices = [i for i, label in enumerate(labels) if label == class_name]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=class_name, alpha=0.75, s=dot_size)
    plt.title("First Two Principal Components", color='cyan')
    plt.xlabel("Principal Component 1", color='cyan')
    plt.ylabel("Principal Component 2", color='cyan')
    plt.legend()
    plt.grid(True, color='gray')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(plt)

def plot_pca_3d(class_names, dot_size):
    combined_features, labels = load_features(class_names)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(combined_features)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    trace = go.Scatter3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=dot_size,
            color=labels_encoded,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='PC1', backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            yaxis=dict(title='PC2', backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            zaxis=dict(title='PC3', backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        ),
        template='plotly_dark',
        width=1000,
        height=800
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

st.title("PCA Visualization")

all_classes = get_all_class_names()
selected_classes = st.multiselect("Select classes", all_classes)

if selected_classes:
    pca_option = st.radio("Select PCA plot", ["2D PCA", "3D PCA"])

    dot_size = st.slider("Select dot size", min_value=1, max_value=100, value=20)

    if st.button("Submit"):
        if pca_option == "2D PCA":
            plot_pca_2d(selected_classes, dot_size)
        elif pca_option == "3D PCA":
            plot_pca_3d(selected_classes, dot_size)
