import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

def load_matlab_data(filename):
    """Load data from MATLAB .mat file."""
    data = loadmat(filename)
    return data

def plot_convergence(result, figsize: tuple = (10, 4)) -> plt.Figure:
    """
    Plot cost and gradient norm histories from an optimization result.
    
    Parameters
    ----------
    result : OptimizationResult
        The result object returned by optimizer.optimize()
    figsize : tuple
        Figure size (width, height) in inches
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (call plt.show() to display)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    iters = np.arange(len(result.cost_history))
    
    # Cost plot
    ax = axes[0]
    ax.semilogy(iters, result.cost_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs Iteration")
    ax.grid(True, alpha=0.3)
    
    # Gradient norms plot
    ax = axes[1]
    ax.semilogy(iters, result.grad_norm_L_history, label=r"$\|\nabla_L\|$")
    ax.semilogy(iters, result.grad_norm_B_history, label=r"$\|\nabla_B\|$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

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