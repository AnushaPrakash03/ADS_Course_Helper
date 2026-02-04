"""
Helper Functions for Notebook 3: Multiple Regression
Course: Introduction to Statistical Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_correlation_heatmap(df, title="Correlation Matrix"):
    """Shows correlation between variables."""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.3f', vmin=-1, vmax=1)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("Values close to ±1: Strong correlation")
    print("Values close to 0: Weak correlation")
    print("⚠️  High correlation between predictors → multicollinearity")
    print("="*70)


def plot_cooks_distance(model, threshold=None):
    """Plots Cook's distance."""
    from statsmodels.stats.outliers_influence import OLSInfluence
    
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    
    if threshold is None:
        threshold = 4 / len(cooks_d)
    
    n = len(cooks_d)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if d > threshold else 'steelblue' for d in cooks_d]
    ax.stem(range(n), cooks_d, markerfmt='o', basefmt=' ')
    
    for i, (d, color) in enumerate(zip(cooks_d, colors)):
        ax.plot(i, d, 'o', color=color, markersize=8, alpha=0.7)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, 
              label=f'Threshold = {threshold:.4f}')
    
    influential_idx = np.where(cooks_d > threshold)[0]
    for idx in influential_idx:
        ax.annotate(f'{idx}', xy=(idx, cooks_d[idx]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color='red', fontweight='bold')
    
    ax.set_xlabel('Observation Index', fontsize=12)
    ax.set_ylabel("Cook's Distance", fontsize=12)
    ax.set_title("Cook's Distance: Influential Points", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print(f"Threshold: {threshold:.4f}")
    print(f"Influential points: {len(influential_idx)}")
    if len(influential_idx) > 0:
        print(f"⚠️  Points: {list(influential_idx)}")
    else:
        print("✅ No highly influential points")
    print("="*70)


def visualize_interaction_effect(df, x1_col, x2_col, y_col, model):
    """Visualizes interaction effects."""
    interaction_term = f'{x1_col}:{x2_col}'
    
    if interaction_term not in model.params:
        print(f"❌ No interaction term in model!")
        return
    
    beta_x1 = model.params[x1_col]
    beta_x2 = model.params[x2_col]
    beta_interaction = model.params[interaction_term]
    intercept = model.params['Intercept']
    
    x1_range = np.linspace(df[x1_col].min(), df[x1_col].max(), 100)
    x2_levels = np.percentile(df[x2_col], [10, 50, 90])
    x2_labels = ['Low', 'Medium', 'High']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for x2_val, label in zip(x2_levels, x2_labels):
        y_pred = intercept + beta_x1 * x1_range + beta_x2 * x2_val + beta_interaction * x1_range * x2_val
        ax.plot(x1_range, y_pred, linewidth=3, label=f'{x2_col}={x2_val:.1f} ({label})')
    
    ax.set_xlabel(f'{x1_col}', fontsize=13)
    ax.set_ylabel(f'Predicted {y_col}', fontsize=13)
    ax.set_title(f'Interaction: {x1_col} × {x2_col}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print(f"Effect of {x1_col} depends on {x2_col}:")
    for x2_val, label in zip(x2_levels, x2_labels):
        effect = beta_x1 + beta_interaction * x2_val
        print(f"   {x2_col}={x2_val:.1f} ({label}): {effect:.4f}")
    
    if beta_interaction > 0:
        print(f"\n✅ Positive synergy (β={beta_interaction:.4f})")
    else:
        print(f"\n⚠️  Negative interaction (β={beta_interaction:.4f})")
    print("="*70)


print("✅ Notebook 3 helpers loaded!")