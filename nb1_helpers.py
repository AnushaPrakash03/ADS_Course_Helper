"""
Helper Functions for Notebook 1: Simple Linear Regression
Course: Introduction to Statistical Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def show_three_slopes():
    """Demonstrates how different slopes fit the same data."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    true_slope = 2
    true_intercept = 5
    noise = np.random.normal(0, 3, 50)
    y = true_intercept + true_slope * x + noise
    
    slopes = [1.5, 2.0, 2.5]
    colors = ['red', 'green', 'blue']
    labels = ['Slope = 1.5 (too flat)', 'Slope = 2.0 (just right)', 'Slope = 2.5 (too steep)']
    
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y, alpha=0.6, s=50, color='black', label='Actual Data', zorder=3)
    
    for slope, color, label in zip(slopes, colors, labels):
        y_line = true_intercept + slope * x
        plt.plot(x, y_line, color=color, linewidth=2.5, label=label, linestyle='--')
    
    plt.xlabel('X (Predictor)', fontsize=13)
    plt.ylabel('Y (Response)', fontsize=13)
    plt.title('Same Data, Different Lines: Which Fits Best?', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("💡 Question: Which line minimizes the vertical distances to the points?")
    print("   (Hint: The green line was fit using least squares!)")


def show_zero_slope():
    """Demonstrates what happens when slope is zero."""
    np.random.seed(123)
    x_flat = np.linspace(0, 10, 50)
    y_mean = 15
    y_flat = y_mean + np.random.normal(0, 2, 50)
    
    slope_actual = np.cov(x_flat, y_flat)[0, 1] / np.var(x_flat)
    intercept_actual = np.mean(y_flat) - slope_actual * np.mean(x_flat)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(x_flat, y_flat, alpha=0.6, s=50, color='darkblue', label='Data')
    axes[0].plot(x_flat, intercept_actual + slope_actual * x_flat, 
                 color='red', linewidth=2.5, linestyle='--', 
                 label=f'Fitted Line (slope ≈ {slope_actual:.3f})')
    axes[0].axhline(y=y_mean, color='green', linewidth=2.5, linestyle='-', 
                    label=f'Horizontal Line (mean = {y_mean})')
    axes[0].set_xlabel('X (Predictor)', fontsize=12)
    axes[0].set_ylabel('Y (Response)', fontsize=12)
    axes[0].set_title('When Slope ≈ 0: X Doesn\'t Help Predict Y', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(x_flat, y_flat, alpha=0.6, s=50, color='darkblue')
    axes[1].axhline(y=y_mean, color='green', linewidth=2.5, linestyle='-', 
                    label='Best Prediction = Mean')
    for i in range(len(x_flat)):
        axes[1].plot([x_flat[i], x_flat[i]], [y_mean, y_flat[i]], 
                    color='red', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('X (Predictor)', fontsize=12)
    axes[1].set_ylabel('Y (Response)', fontsize=12)
    axes[1].set_title('Residuals: Just Random Scatter Around Mean', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("🔍 Key Insight: When β₁ ≈ 0, knowing X tells us nothing about Y!")
    print(f"   Best prediction for Y is always ȳ = {y_mean}, regardless of X value.")


def show_anscombe_statistics():
    """Prints Anscombe's Quartet summary statistics."""
    anscombe_data = {
        'x1': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y1': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        'x2': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y2': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        'x3': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y3': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        'x4': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        'y4': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    }
    
    df_anscombe = pd.DataFrame(anscombe_data)
    
    print("=" * 70)
    print("ANSCOMBE'S QUARTET - SUMMARY STATISTICS")
    print("=" * 70)
    
    datasets = [('I', 'x1', 'y1'), ('II', 'x2', 'y2'), 
                ('III', 'x3', 'y3'), ('IV', 'x4', 'y4')]
    
    for name, x_col, y_col in datasets:
        x = df_anscombe[x_col]
        y = df_anscombe[y_col]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        print(f"\nDataset {name}:")
        print(f"  Mean of X: {x.mean():.2f}")
        print(f"  Mean of Y: {y.mean():.2f}")
        print(f"  Variance of X: {x.var():.2f}")
        print(f"  Variance of Y: {y.var():.2f}")
        print(f"  Correlation: {r_value:.3f}")
        print(f"  Regression: y = {intercept:.2f} + {slope:.2f}x")
    
    print("\n" + "=" * 70)
    print("👆 LOOK! All four datasets have nearly IDENTICAL statistics!")
    print("=" * 70)
    
    return df_anscombe


def plot_anscombe_quartet():
    """Creates the famous 2x2 plot."""
    anscombe_data = {
        'x1': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y1': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        'x2': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y2': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        'x3': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y3': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        'x4': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        'y4': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    }
    
    df_anscombe = pd.DataFrame(anscombe_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Anscombe's Quartet: Same Stats, Different Stories", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    datasets_plot = [
        ('I', 'x1', 'y1', 'Actually Linear', axes[0, 0]),
        ('II', 'x2', 'y2', 'Non-linear (Curved!)', axes[0, 1]),
        ('III', 'x3', 'y3', 'Linear + Outlier', axes[1, 0]),
        ('IV', 'x4', 'y4', 'Leverage Point Illusion', axes[1, 1])
    ]
    
    for name, x_col, y_col, description, ax in datasets_plot:
        x = df_anscombe[x_col]
        y = df_anscombe[y_col]
        
        ax.scatter(x, y, s=80, alpha=0.7, color='darkblue', edgecolors='black', linewidth=1.5)
        
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, 'r--', linewidth=2.5, label=f'y = {intercept:.1f} + {slope:.2f}x')
        
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_title(f'Dataset {name}: {description}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(3, 20)
        ax.set_ylim(3, 13)
        
        stats_text = f'r = {r_value:.3f}\nmean(x) = {x.mean():.1f}\nmean(y) = {y.mean():.1f}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print("THE LESSON:")
    print("=" * 70)
    print("📊 Dataset I:  Perfect for linear regression (actually linear)")
    print("📈 Dataset II: LINEAR REGRESSION IS WRONG! This is curved/quadratic")
    print("⚠️  Dataset III: One outlier ruins everything - investigate it!")
    print("🎪 Dataset IV:  One extreme point creates fake relationship")
    print("=" * 70)
    print("✅ ALWAYS PLOT YOUR DATA BEFORE FITTING MODELS!")


def create_advertising_dataset():
    """Creates and returns the advertising dataset."""
    np.random.seed(42)
    n = 200
    
    tv = np.random.uniform(0, 300, n)
    radio = np.random.uniform(0, 50, n)
    newspaper = np.random.uniform(0, 120, n)
    
    sales = (7 + 0.047 * tv + 0.18 * radio + 0.001 * newspaper + 
             np.random.normal(0, 1.8, n))
    
    df_adv = pd.DataFrame({
        'TV': tv,
        'Radio': radio,
        'Newspaper': newspaper,
        'Sales': sales
    })
    
    print("=" * 60)
    print("ADVERTISING DATASET CREATED")
    print("=" * 60)
    print(f"Observations: {len(df_adv)}")
    print(f"\nFirst 5 rows:")
    print(df_adv.head())
    print(f"\nSummary:")
    print(df_adv.describe().round(2))
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(['TV', 'Radio', 'Newspaper']):
        axes[i].scatter(df_adv[col], df_adv['Sales'], alpha=0.5)
        axes[i].set_xlabel(f'{col} Budget ($1000s)')
        axes[i].set_ylabel('Sales (1000 units)')
        axes[i].set_title(f'{col} vs Sales')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✅ Dataset ready!")
    
    return df_adv


def visualize_residual_properties():
    """Visualizes residual properties."""
    np.random.seed(42)
    n = 30
    x = np.linspace(0, 10, n)
    y = 5 + 2*x + np.random.normal(0, 3, n)
    
    slope, intercept = np.polyfit(x, y, 1)
    y_fitted = intercept + slope * x
    residuals = y - y_fitted
    
    positive_residuals = residuals > 0
    negative_residuals = residuals < 0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    ax1.plot(x, y_fitted, 'b-', linewidth=2.5, label='Regression Line', zorder=2)
    ax1.scatter(x[positive_residuals], y[positive_residuals], 
               s=80, color='green', edgecolors='black', linewidth=1.5,
               label='Above line (positive)', zorder=3, alpha=0.7)
    ax1.scatter(x[negative_residuals], y[negative_residuals], 
               s=80, color='red', edgecolors='black', linewidth=1.5,
               label='Below line (negative)', zorder=3, alpha=0.7)
    
    for i in range(n):
        color = 'green' if residuals[i] > 0 else 'red'
        ax1.plot([x[i], x[i]], [y_fitted[i], y[i]], 
                color=color, linewidth=2, alpha=0.6, zorder=1)
    
    ax1.set_xlabel('X (Predictor)', fontsize=13)
    ax1.set_ylabel('Y (Response)', fontsize=13)
    ax1.set_title('Residuals: Vertical Distances from Line', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    colors = ['green' if r > 0 else 'red' for r in residuals]
    ax2.bar(range(n), residuals, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='blue', linewidth=2.5, linestyle='--', label='Zero line')
    ax2.set_xlabel('Observation Index', fontsize=13)
    ax2.set_ylabel('Residual (eᵢ = yᵢ - ŷᵢ)', fontsize=13)
    ax2.set_title('Individual Residuals: They Balance Out!', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("RESIDUAL PROPERTIES")
    print("=" * 70)
    print(f"\n✅ Property 1: Σeᵢ = {np.sum(residuals):.10f} ≈ 0")
    print(f"✅ Property 2: Σ(xᵢ×eᵢ) = {np.sum(x * residuals):.10f} ≈ 0")
    
    n_pos = np.sum(positive_residuals)
    sum_pos = np.sum(residuals[positive_residuals])
    sum_neg = np.sum(residuals[negative_residuals])
    
    print(f"\n📊 Balance:")
    print(f"   Positive: {n_pos} points, sum = {sum_pos:.2f}")
    print(f"   Negative: {n-n_pos} points, sum = {sum_neg:.2f}")
    print(f"   Total: {sum_pos + sum_neg:.10f}")
    print("=" * 70)


print("✅ Notebook 1 helpers loaded!")