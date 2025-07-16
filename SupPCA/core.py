import numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions, ConjugateGradient
from pymanopt import Problem
import pymanopt.function

class LSPCA:
    def __init__(self, n_components, max_iter=500, tol=1e-6, verbose=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.components_ = None
        self.explained_variance_ = None
        
    def fit(self, X):
        """
        Fit the LSPCA model to data X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix (or Gram matrix for efficiency)
        if n_samples > n_features:
            # Standard covariance
            C = (X_centered.T @ X_centered) / n_samples
        else:
            # Use Gram matrix trick
            G = (X_centered @ X_centered.T) / n_samples
            
        # Define manifold
        manifold = Stiefel(n_features, self.n_components)
        
        # Define cost function and gradient
        @pymanopt.function.numpy(manifold)
        def cost(U):
            # Maximize trace(U^T C U) => minimize -trace(U^T C U)
            return -np.trace(U.T @ C @ U)
        
        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(U):
            # Gradient of -trace(U^T C U) is -2CU
            return -2 * C @ U
        
        # Alternative: you can also define the Riemannian gradient directly
        # def riemannian_gradient(U):
        #     egrad = -2 * C @ U
        #     # Project to tangent space
        #     return manifold.proj(U, egrad)
        
        # Create problem
        problem = Problem(
            manifold=manifold,
            cost=cost,
            euclidean_gradient=euclidean_gradient
        )
        
        # Choose optimizer
        optimizer = TrustRegions(
            max_iterations=self.max_iter,
            min_gradient_norm=self.tol
        )
        
        # Initial point: random orthonormal matrix
        U0 = np.linalg.qr(np.random.randn(n_features, self.n_components))[0]
        
        # Optimize
        result = optimizer.run(problem, initial_point=U0)
        
        # Store results
        self.components_ = result.point
        
        # Compute explained variance
        X_transformed = X_centered @ self.components_
        self.explained_variance_ = np.var(X_transformed, axis=0)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(np.var(X_centered, axis=0))
        )
        
        return self
    
    def transform(self, X):
        """Transform data using the learned components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X):
        """Fit the model and transform the data."""
        self.fit(X)
        return self.transform(X)