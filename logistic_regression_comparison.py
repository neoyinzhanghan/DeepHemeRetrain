import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Set Seaborn dark theme
sns.set_theme(style="darkgrid", palette="deep")

# Define directories
features_dir = "/Users/neo/Documents/DATA/pooled_deepheme_data_features"
features_dir_train = os.path.join(features_dir, "train")
features_dir_val = os.path.join(features_dir, "val")
features_dir_test = os.path.join(features_dir, "test")


def get_all_class_names():
    """
    Get the names of all classes in the features directory.

    Returns:
    list: A list of strings, where each string is the name of a class.
    """
    class_names = [
        d
        for d in os.listdir(features_dir_train)
        if os.path.isdir(os.path.join(features_dir_train, d))
    ]
    return class_names


def load_data(features_dir, class_name):
    class_dir = os.path.join(features_dir, class_name)
    files = [
        os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".npy")
    ]
    data = [np.load(f) for f in files]
    return np.vstack(data)


def get_labels(data, class_label):
    return np.full(data.shape[0], class_label)


def evaluate_model(model, X, y, best_threshold=None):
    y_pred_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    if best_threshold is None:
        best_threshold = 0.5

    y_pred = (y_pred_prob >= best_threshold).astype(int)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision_cls = precision_score(y, y_pred, average=None)
    recall_cls = recall_score(y, y_pred, average=None)
    return (
        fpr,
        tpr,
        precision,
        recall,
        roc_auc,
        pr_auc,
        accuracy,
        f1,
        y_pred_prob,
        precision_cls,
        recall_cls,
    )


def plot_curves(
    fpr,
    tpr,
    precision,
    recall,
    roc_auc,
    pr_auc,
    dataset_name,
    best_threshold,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="black")

    # ROC Curve
    axes[0].plot(
        fpr, tpr, color="#00FF00", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    axes[0].plot([0, 1], [0, 1], color="#004000", lw=2, linestyle="--")
    axes[0].set_xlabel("False Positive Rate", color="white")
    axes[0].set_ylabel("True Positive Rate", color="white")
    axes[0].set_title(f"ROC Curve ({dataset_name})", color="white")
    axes[0].legend(
        loc="lower right", facecolor="black", edgecolor="green", labelcolor="white"
    )
    axes[0].axhline(y=best_threshold, color="#FF0000", linestyle="--")
    axes[0].axvline(x=best_threshold, color="#FF0000", linestyle="--")
    axes[0].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_facecolor("black")
    axes[0].tick_params(axis="x", colors="white")
    axes[0].tick_params(axis="y", colors="white")

    # Precision-Recall Curve
    axes[1].plot(
        recall,
        precision,
        color="#00FF00",
        lw=2,
        label=f"PR curve (area = {pr_auc:.2f})",
    )
    axes[1].set_xlabel("Recall", color="white")
    axes[1].set_ylabel("Precision", color="white")
    axes[1].set_title(f"Precision-Recall Curve ({dataset_name})", color="white")
    axes[1].legend(
        loc="lower left", facecolor="black", edgecolor="green", labelcolor="white"
    )
    axes[1].axhline(y=best_threshold, color="#FF0000", linestyle="--")
    axes[1].axvline(x=best_threshold, color="#FF0000", linestyle="--")
    axes[1].grid(color="gray", linestyle="--", linewidth=0.5)
    axes[1].set_facecolor("black")
    axes[1].tick_params(axis="x", colors="white")
    axes[1].tick_params(axis="y", colors="white")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Matrix-Themed Curves", color="white", fontsize=16, weight="bold")

    return fig


def get_pairwise_lr_roc(class_name_1, class_name_2):
    # Load train data
    X_train_1 = load_data(features_dir_train, class_name_1)
    X_train_2 = load_data(features_dir_train, class_name_2)
    X_train = np.vstack([X_train_1, X_train_2])
    y_train = np.hstack([get_labels(X_train_1, 0), get_labels(X_train_2, 1)])

    # Load val data
    X_val_1 = load_data(features_dir_val, class_name_1)
    X_val_2 = load_data(features_dir_val, class_name_2)
    X_val = np.vstack([X_val_1, X_val_2])
    y_val = np.hstack([get_labels(X_val_1, 0), get_labels(X_val_2, 1)])

    # Load test data
    X_test_1 = load_data(features_dir_test, class_name_1)
    X_test_2 = load_data(features_dir_test, class_name_2)
    X_test = np.vstack([X_test_1, X_test_2])
    y_test = np.hstack([get_labels(X_test_1, 0), get_labels(X_test_2, 1)])

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate on validation set to determine best threshold
    _, _, _, _, _, _, _, _, y_val_pred_prob, _, _ = evaluate_model(model, X_val, y_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]

    # Evaluate on train set
    (
        fpr_train,
        tpr_train,
        precision_train,
        recall_train,
        roc_auc_train,
        pr_auc_train,
        accuracy_train,
        f1_train,
        _,
        precision_cls_train,
        recall_cls_train,
    ) = evaluate_model(model, X_train, y_train, best_threshold)

    # Evaluate on validation set
    (
        fpr_val,
        tpr_val,
        precision_val,
        recall_val,
        roc_auc_val,
        pr_auc_val,
        accuracy_val,
        f1_val,
        _,
        precision_cls_val,
        recall_cls_val,
    ) = evaluate_model(model, X_val, y_val, best_threshold)

    # Evaluate on test set
    (
        fpr_test,
        tpr_test,
        precision_test,
        recall_test,
        roc_auc_test,
        pr_auc_test,
        accuracy_test,
        f1_test,
        _,
        precision_cls_test,
        recall_cls_test,
    ) = evaluate_model(model, X_test, y_test, best_threshold)

    # Plot all curves on the same page
    fig_train = plot_curves(
        fpr_train,
        tpr_train,
        precision_train,
        recall_train,
        roc_auc_train,
        pr_auc_train,
        "Train",
        best_threshold,
    )

    fig_val = plot_curves(
        fpr_val,
        tpr_val,
        precision_val,
        recall_val,
        roc_auc_val,
        pr_auc_val,
        "Validation",
        best_threshold,
    )

    fig_test = plot_curves(
        fpr_test,
        tpr_test,
        precision_test,
        recall_test,
        roc_auc_test,
        pr_auc_test,
        "Test",
        best_threshold,
    )

    return (
        (
            fig_train,
            accuracy_train,
            f1_train,
            roc_auc_train,
            precision_cls_train,
            recall_cls_train,
        ),
        (fig_val, accuracy_val, f1_val, roc_auc_val, precision_cls_val, recall_cls_val),
        (
            fig_test,
            accuracy_test,
            f1_test,
            roc_auc_test,
            precision_cls_test,
            recall_cls_test,
        ),
    )


# Streamlit app
st.title("Pairwise Logistic Regression Performance")

class_names = get_all_class_names()

class_name_1 = st.selectbox("Select first class", class_names)
class_name_2 = st.selectbox("Select second class", class_names)

if st.button("Evaluate"):
    train_results, val_results, test_results = get_pairwise_lr_roc(
        class_name_1, class_name_2
    )

    for results, dataset in zip(
        [train_results, val_results, test_results], ["Train", "Validation", "Test"]
    ):
        fig, acc, f1, auc_score, prec, rec = results
        st.pyplot(fig)
        st.markdown(f"### {dataset} Set Performance")
        st.markdown(f"**Accuracy:** {acc:.2f}")
        st.markdown(f"**F1 Score:** {f1:.2f}")
        st.markdown(f"**AUC Score:** {auc_score:.2f}")
