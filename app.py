import streamlit as st
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

st.title("Wellcome to learn Decision tree")

# Sidebar for Hyperparameters
st.sidebar.header("Decision Tree Hyperparameters")
max_depth = st.sidebar.slider("Max Depth", 1, 20, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
max_features = st.sidebar.slider("Max Features (fraction)", 1, 10, 10) / 10

# Generate synthetic dataset
@st.cache
def generate_dataset():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

X, y = generate_dataset()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Decision Tree Model
clf = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features
)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Display Results
st.subheader("Accuracy Results")
st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Test Accuracy: {test_acc:.2f}")

# Plot decision boundary
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    return plt

st.subheader("Decision Boundary")
fig = plot_decision_boundary(clf, X, y)
st.pyplot(fig)

# Explain Underfitting and Overfitting
st.subheader("Analysis of Overfitting and Underfitting")
if max_depth <= 2 or min_samples_split >= len(X_train):
    st.write("**Underfitting**: The model is too simple to capture patterns in the data.")
elif max_depth > 15 and min_samples_leaf == 1:
    st.write("**Overfitting**: The model is too complex and fits the noise in the data.")
else:
    st.write("The model is well-balanced.")
