"""
Helper Functions for Notebook 2: Inference, Uncertainty & Model Fit
Course: Introduction to Statistical Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, f
from scipy import stats as sp_stats
from scipy.interpolate import make_interp_spline

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def demonstrate_sampling_variability(n_samples=5, sample_size=50, seed=42):
    """Shows how slope estimates vary across samples."""
    np.random.seed(seed)
    population_size = 10000
    x_pop = np.random.uniform(0, 10, population_size)
    true_beta0, true_beta1, true_sigma = 5, 2, 3
    y_pop = true_beta0 + true_beta1 * x_pop + np.random.normal(0, true_sigma, population_size)
    
    slopes = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i in range(n_samples):
        idx = np.random.choice(population_size, sample_size, replace=False)
        x_sample = x_pop[idx]
        y_sample = y_pop[idx]
        
        slope, intercept = np.polyfit(x_sample, y_sample, 1)
        slopes.append(slope)
        
        axes[i].scatter(x_sample, y_sample, alpha=0.6, s=50, color='steelblue')
        x_line = np.linspace(0, 10, 100)
        axes[i].plot(x_line, intercept + slope*x_line, 'r--', linewidth=2.5, 
                    label=f'β₁ = {slope:.3f}')
        axes[i].plot(x_line, true_beta0 + true_beta1*x_line, 'g-', linewidth=2,
                    label=f'True: {true_beta1}', alpha=0.7)
        axes[i].set_title(f'Sample {i+1}', fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-0.5, 10.5)
        axes[i].set_ylim(0, 30)
    
    axes[5].hist(slopes, bins=10, color='coral', alpha=0.7, edgecolor='black')
    axes[5].axvline(true_beta1, color='green', linewidth=3, label=f'True={true_beta1}')
    axes[5].axvline(np.mean(slopes), color='red', linewidth=3, linestyle='--', 
                   label=f'Mean={np.mean(slopes):.3f}')
    axes[5].set_xlabel('Estimated Slope')
    axes[5].set_title('Distribution', fontweight='bold')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print(f"True: {true_beta1}, Mean: {np.mean(slopes):.3f}, SD: {np.std(slopes, ddof=1):.3f}")
    print("="*70)


def visualize_r_squared():
    """Shows R² decomposition."""
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 10, n)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    scenarios = [
        (5 + 2*x + np.random.normal(0, 2, n), "High R²", 0),
        (5 + 2*x + np.random.normal(0, 6, n), "Low R²", 1)
    ]
    
    for y, title, row in scenarios:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x
        y_mean = np.mean(y)
        
        SS_tot = np.sum((y - y_mean)**2)
        SS_reg = np.sum((y_pred - y_mean)**2)
        SS_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (SS_res / SS_tot)
        
        axes[row, 0].scatter(x, y, alpha=0.6, s=50, color='steelblue')
        axes[row, 0].axhline(y_mean, color='green', linewidth=3)
        for i in range(0, n, 5):
            axes[row, 0].plot([x[i], x[i]], [y_mean, y[i]], 'purple', alpha=0.3, linewidth=1.5)
        axes[row, 0].set_title(f'{title}\nTotal (SS_tot={SS_tot:.1f})', fontweight='bold')
        axes[row, 0].legend(['Mean', 'Data'])
        axes[row, 0].grid(True, alpha=0.3)
        
        axes[row, 1].scatter(x, y, alpha=0.3, s=50, color='gray')
        axes[row, 1].plot(x, y_pred, 'r-', linewidth=3)
        axes[row, 1].axhline(y_mean, color='green', linewidth=2, linestyle='--', alpha=0.5)
        for i in range(0, n, 5):
            axes[row, 1].plot([x[i], x[i]], [y_mean, y_pred[i]], 'orange', alpha=0.5, linewidth=2)
        axes[row, 1].set_title(f'Explained (SS_reg={SS_reg:.1f})', fontweight='bold')
        axes[row, 1].grid(True, alpha=0.3)
        
        axes[row, 2].scatter(x, y, alpha=0.6, s=50, color='steelblue')
        axes[row, 2].plot(x, y_pred, 'r-', linewidth=3)
        for i in range(0, n, 5):
            axes[row, 2].plot([x[i], x[i]], [y_pred[i], y[i]], 'red', alpha=0.4, linewidth=1.5)
        axes[row, 2].set_title(f'Unexplained (SS_res={SS_res:.1f})\nR²={r_squared:.3f}', fontweight='bold')
        axes[row, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("R² = 1 - (SS_res / SS_tot)")
    print("="*70)


def create_diagnostic_plots(x, y, title="Diagnostics"):
    """Creates diagnostic plots."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    residuals = y - y_pred
    residual_std = residuals / np.std(residuals)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue')
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    sp_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(residual_std)), alpha=0.6, s=50, color='steelblue')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Std Residuals|')
    axes[1, 0].set_title('Scale-Location', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(range(len(residuals)), residual_std, alpha=0.6, s=50, color='steelblue')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1.5)
    axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1.5)
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel('Std Residuals')
    axes[1, 1].set_title('Residuals vs Order', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("✓ Check for patterns, normality, constant variance")
    print("="*70)


def demonstrate_good_vs_bad_residuals():
    """Shows good vs bad residual patterns."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    
    scenarios = [
        (5 + 2*x + np.random.normal(0, 2, n), "✅ Good: Random"),
        (5 + 2*x + 0.5*x**2 + np.random.normal(0, 2, n), "❌ Bad: Non-linear"),
        (5 + 2*x + np.random.normal(0, 0.2*x, n), "❌ Bad: Heteroscedasticity"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, (y, title) in enumerate(scenarios):
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x
        residuals = y - y_pred
        
        axes[0, idx].scatter(x, y, alpha=0.5, s=40, color='steelblue')
        axes[0, idx].plot(x, y_pred, 'r-', linewidth=2.5)
        axes[0, idx].set_title(title, fontweight='bold', fontsize=11)
        axes[0, idx].grid(True, alpha=0.3)
        
        axes[1, idx].scatter(y_pred, residuals, alpha=0.6, s=40, color='steelblue')
        axes[1, idx].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, idx].set_xlabel('Fitted Values')
        axes[1, idx].set_ylabel('Residuals')
        axes[1, idx].grid(True, alpha=0.3)
        
        if "Good" in title:
            axes[0, idx].set_facecolor('#e8f5e9')
            axes[1, idx].set_facecolor('#e8f5e9')
        else:
            axes[0, idx].set_facecolor('#ffebee')
            axes[1, idx].set_facecolor('#ffebee')
    
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("✅ GOOD: Random scatter")
    print("❌ BAD: Patterns or funnel shapes")
    print("="*70)


print("✅ Notebook 2 helpers loaded!")