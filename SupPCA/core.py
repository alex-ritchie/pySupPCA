import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass, field

from utils import plot_convergence
import matplotlib.pyplot as plt

# =============================================================================
# Stepsize schedules
# =============================================================================

def constant_stepsize(alpha: float) -> Callable[[int], float]:
    """Returns a constant stepsize function."""
    def schedule(k: int) -> float:
        return alpha
    return schedule


def diminishing_stepsize(alpha: float, decay: float = 1.0) -> Callable[[int], float]:
    """Returns a diminishing stepsize: alpha / (1 + decay * k)"""
    def schedule(k: int) -> float:
        return alpha / (1.0 + decay * k)
    return schedule


# =============================================================================
# Optimization result container
# =============================================================================

@dataclass
class OptimizationResult:
    L: np.ndarray
    B: np.ndarray
    cost_history: list = field(default_factory=list)
    grad_norm_L_history: list = field(default_factory=list)
    grad_norm_B_history: list = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


# =============================================================================
# Abstract base class for alternating optimizers
# =============================================================================

class AlternatingOptimizer(ABC):
    """
    Abstract base class for alternating minimization over two blocks (L, B).
    
    Subclasses must implement:
        - cost(L, B) -> float
        - grad_L(L, B) -> np.ndarray
        - grad_B(L, B) -> np.ndarray
    
    Optionally override:
        - update_L(L, B, stepsize) -> np.ndarray
        - update_B(L, B, stepsize) -> np.ndarray
    """
    
    def __init__(
        self,
        L_init: np.ndarray,
        B_init: np.ndarray,
        stepsize_L: Callable[[int], float] = constant_stepsize(1e-3),
        stepsize_B: Callable[[int], float] = constant_stepsize(1e-3),
    ):
        self.L = L_init.copy()
        self.B = B_init.copy()
        self.stepsize_L = stepsize_L
        self.stepsize_B = stepsize_B
        self.result = None
    
    @abstractmethod
    def cost(self, L: np.ndarray, B: np.ndarray) -> float:
        """Compute the objective function value."""
        pass
    
    @abstractmethod
    def grad_L(self, L: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gradient with respect to L, holding B fixed."""
        pass
    
    @abstractmethod
    def grad_B(self, L: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gradient with respect to B, holding L fixed."""
        pass
    
    def update_L(self, L: np.ndarray, B: np.ndarray, stepsize: float) -> np.ndarray:
        """Update rule for L. Default: gradient descent."""
        return L - stepsize * self.grad_L(L, B)
    
    def update_B(self, L: np.ndarray, B: np.ndarray, stepsize: float) -> np.ndarray:
        """Update rule for B. Default: gradient descent."""
        return B - stepsize * self.grad_B(L, B)
    
    def optimize(
        self,
        max_iters: int = 1000,
        tol: float = 1e-8,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> OptimizationResult:
        """Run the alternating optimization."""
        
        cost_history = []
        grad_norm_L_history = []
        grad_norm_B_history = []
        
        cost_prev = np.inf
        converged = False
        
        for k in range(max_iters):
            # Compute current cost and gradients
            c = self.cost(self.L, self.B)
            gL = self.grad_L(self.L, self.B)
            gB = self.grad_B(self.L, self.B)
            
            cost_history.append(c)
            grad_norm_L_history.append(np.linalg.norm(gL))
            grad_norm_B_history.append(np.linalg.norm(gB))
            
            # Check convergence
            if np.abs(cost_prev - c) < tol:
                if verbose:
                    print(f"Converged at iteration {k}")
                converged = True
                break
            cost_prev = c
            
            # Logging
            if verbose and k % log_interval == 0:
                print(f"Iter {k:5d} | Cost: {c:.6e} | ||grad_L||: {grad_norm_L_history[-1]:.3e} | ||grad_B||: {grad_norm_B_history[-1]:.3e}")
            
            # Alternating updates
            alpha_L = self.stepsize_L(k)
            alpha_B = self.stepsize_B(k)
            
            self.L = self.update_L(self.L, self.B, alpha_L)
            self.B = self.update_B(self.L, self.B, alpha_B)
        
        self.result = OptimizationResult(
            L=self.L,
            B=self.B,
            cost_history=cost_history,
            grad_norm_L_history=grad_norm_L_history,
            grad_norm_B_history=grad_norm_B_history,
            iterations=k + 1,
            converged=converged,
        )
        
        return self.result


# =============================================================================
# Numba-accelerated kernels (defined at module level for compilation)
# =============================================================================

@njit
def _sup_pca_cost(L: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float) -> float:
    Yerr = Y - X @ (L @ B)
    Xerr = X - (X @ L) @ L.T
    return 0.5*(1-lam)*np.sum(Yerr ** 2) + 0.5*lam*np.sum(Xerr ** 2)


@njit
def _sup_pca_egrad_L(L: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    XtXL = X.T @ (X @ L)
    return (lam - 1)*(L.T @ X.T) @ (Y - X @ (L @ B)) @ B.T - lam*(2*XtXL - (XtXL @ L.T) @ L - L @ (L.T @ XtXL))

@njit
def _sup_pca_rgrad_L(L: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    XtXL = X.T @ (X @ L)
    quick_egrad = (lam - 1)*(L.T @ X.T) @ (Y - X @ (L @ B)) @ B.T - lam*(XtXL)
    return quick_egrad - L @ (L.T @ quick_egrad)

@njit
def _sup_pca_grad_B(L: np.ndarray, B: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    return (lam - 1)*(L.T @ X.T) @ (Y - X @ (L @ B))


# =============================================================================
# Concrete implementation: Matrix Factorization
# =============================================================================

class MatrixFactorization(AlternatingOptimizer):
    """
    Solves: min_{L, B} 0.5 * ||X - L @ B.T||_F^2
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lam: float,
        L_init: np.ndarray,
        B_init: np.ndarray,
        stepsize_L: Callable[[int], float] = constant_stepsize(1e-3),
        stepsize_B: Callable[[int], float] = constant_stepsize(1e-3),
    ):
        super().__init__(L_init, B_init, stepsize_L, stepsize_B)
        self.X = X
        self.Y = Y
        self.lam = lam
    
    def cost(self, L: np.ndarray, B: np.ndarray) -> float:
        return _sup_pca_cost(L, B, self.X, self.Y, self.lam)
    
    def grad_L(self, L: np.ndarray, B: np.ndarray) -> np.ndarray:
        return _sup_pca_egrad_L(L, B, self.X, self.Y, self.lam)
        # return _sup_pca_rgrad_L(L, B, self.X)
    
    def grad_B(self, L: np.ndarray, B: np.ndarray) -> np.ndarray:
        return _sup_pca_grad_B(L, B, self.X, self.Y, self.lam)


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Problem dimensions
    n, p, q, r = 100, 3, 1, 1
    
    # Generate synthetic data
    L_true = np.random.randn(p, r)
    B_true = np.random.randn(r, q)
    X = np.random.randn(n, p) @ np.diag([8, 4, 2])
    Y = X @ L_true @ B_true + np.random.randn(n, q)
    
    # Initialize
    L_init = np.random.randn(p, r)
    B_init = np.random.randn(r, q)
    lam = 0.9
    
    # Create optimizer and run
    optimizer = MatrixFactorization(
        X=X,
        Y=Y,
        lam=lam,
        L_init=L_init,
        B_init=B_init,
        # stepsize_L=constant_stepsize(1e-4),
        # stepsize_B=constant_stepsize(1e-4),
        stepsize_L=diminishing_stepsize(1e-4, 0.01),
        stepsize_B=diminishing_stepsize(1e-4, 0.01),
    )
    
    result = optimizer.optimize(
        max_iters=5000,
        tol=1e-10,
        verbose=True,
        log_interval=10,
    )
    
    print(f"\nFinished in {result.iterations} iterations")
    print(f"Converged: {result.converged}")
    print(f"Final cost: {result.cost_history[-1]:.6e}")
    
    # Plot convergence
    fig = plot_convergence(result)
    plt.show()