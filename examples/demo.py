import numpy as np
import sys
sys.path.append('..')

from lspca.core import LSPCA
from lspca.utils import plot_convergence, plot_explained_variance

# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 1000, 50
n_components = 10

# Create data with some structure
true_components = np.random.randn(n_features, n_components)
true_components, _ = np.linalg.qr(true_components)
coefficients = np.random.randn(n_samples, n_components)
X = coefficients @ true_components.T + 0.1 * np.random.randn(n_samples, n_features)

# Fit LSPCA
print("Fitting LSPCA model...")
lspca = LSPCA(n_components=n_components, max_iter=100, verbose=True)
X_transformed = lspca.fit_transform(X)

print(f"\nOriginal data shape: {X.shape}")
print(f"Transformed data shape: {X_transformed.shape}")
print(f"Explained variance ratio: {lspca.explained_variance_ratio_}")

# Plot results
plot_explained_variance(lspca)

# Compare with standard PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")