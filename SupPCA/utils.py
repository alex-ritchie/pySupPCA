import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

def load_matlab_data(filename):
    """Load data from MATLAB .mat file."""
    data = loadmat(filename)
    return data

def plot_convergence(optimizer_result):
    """Plot optimization convergence."""
    if hasattr(optimizer_result, 'iterations'):
        iterations = optimizer_result.iterations
        costs = [it['cost'] for it in iterations if 'cost' in it]
        
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('LSPCA Convergence')
        plt.grid(True)
        plt.show()

def plot_explained_variance(lspca_model):
    """Plot explained variance by component."""
    plt.figure(figsize=(10, 6))
    components = range(1, len(lspca_model.explained_variance_) + 1)
    
    plt.subplot(1, 2, 1)
    plt.bar(components, lspca_model.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained by Each Component')
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(lspca_model.explained_variance_ratio_)
    plt.plot(components, cumsum, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()