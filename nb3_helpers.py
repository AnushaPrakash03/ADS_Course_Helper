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
    
    print("=" * 70)
    print("INTERPRETING CORRELATIONS")
    print("=" * 70)
    print("• Values close to +1: Strong positive correlation")
    print("• Values close to -1: Strong negative correlation")
    print("• Values close to 0: Weak/no correlation")
    print("\n⚠️  High correlation between predictors → multicollinearity!")
    print("=" * 70)


def plot_cooks_distance(model, threshold=None):
    """Plots Cook's distance to identify influential points."""
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
    ax.set_title("Cook's Distance: Detecting Influential Points", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("COOK'S DISTANCE ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {threshold:.4f}")
    print(f"Influential points detected: {len(influential_idx)}")
    
    if len(influential_idx) > 0:
        print(f"\n⚠️  Observations: {list(influential_idx)}")
        print("   Action items:")
        print("   1. Check if they're data errors")
        print("   2. Analyze with and without them")
        print("   3. Report sensitivity")
    else:
        print("\n✅ No highly influential points detected")
    
    print("=" * 70)


def visualize_tv_radio_interaction(df, model):
    """Visualizes TV × Radio interaction effect."""
    
    interaction_term = 'TV:Radio'
    
    if interaction_term not in model.params:
        print(f"❌ No interaction term in model!")
        print("   Fit with: Sales ~ TV + Radio + TV:Radio")
        return
    
    beta_tv = model.params['TV']
    beta_radio = model.params['Radio']
    beta_interaction = model.params[interaction_term]
    beta_0 = model.params['Intercept']
    
    # Create TV range
    tv_range = np.linspace(df['TV'].min(), df['TV'].max(), 100)
    
    # Show effect at different Radio levels
    radio_levels = np.percentile(df['Radio'], [10, 50, 90])
    radio_labels = ['Low Radio ($' + f'{r:.0f}' + 'k)' for r in radio_levels]
    colors = ['red', 'green', 'blue']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for radio_val, label, color in zip(radio_levels, radio_labels, colors):
        sales_pred = (beta_0 + 
                     beta_tv * tv_range + 
                     beta_radio * radio_val + 
                     beta_interaction * tv_range * radio_val)
        
        ax.plot(tv_range, sales_pred, linewidth=3, label=label, color=color)
    
    ax.set_xlabel('TV Advertising Budget ($1000s)', fontsize=13)
    ax.set_ylabel('Predicted Sales (1000 units)', fontsize=13)
    ax.set_title('TV × Radio Interaction Effect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, title='Radio Spending Level')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("TV × RADIO INTERACTION")
    print("=" * 70)
    print(f"\nβ₃ (interaction) = {beta_interaction:.6f}")
    print(f"\n📊 Marginal effect of TV at different Radio levels:")
    
    for radio_val, label in zip(radio_levels, radio_labels):
        effect = beta_tv + beta_interaction * radio_val
        print(f"   {label}: ${effect:.4f} per $1k TV")
    
    if beta_interaction > 0:
        print(f"\n✅ Positive synergy!")
        print(f"   TV becomes MORE effective as Radio spending increases")
    else:
        print(f"\n⚠️  Negative interaction")
        print(f"   TV becomes LESS effective as Radio spending increases")
    
    print("=" * 70)


def visualize_income_student_interaction(credit_df, model):
    """Visualizes Income × Student interaction."""
    
    interaction_term = 'Income:Student[T.Yes]'
    
    if interaction_term not in model.params:
        print("❌ No interaction in model!")
        return
    
    beta_income = model.params['Income']
    beta_student = model.params['Student[T.Yes]']
    beta_interaction = model.params[interaction_term]
    beta_0 = model.params['Intercept']
    p_interaction = model.pvalues[interaction_term]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Separate data by student status
    students = credit_df[credit_df['Student'] == 'Yes']
    non_students = credit_df[credit_df['Student'] == 'No']
    
    # Scatter plots
    ax.scatter(non_students['Income'], non_students['Balance'], 
              alpha=0.4, s=40, color='blue', label='Non-Students')
    ax.scatter(students['Income'], students['Balance'], 
              alpha=0.4, s=40, color='red', label='Students')
    
    # Regression lines
    income_range = np.linspace(credit_df['Income'].min(), credit_df['Income'].max(), 100)
    
    # Non-student line
    y_non_student = beta_0 + beta_income * income_range
    
    # Student line
    y_student = (beta_0 + beta_student + 
                (beta_income + beta_interaction) * income_range)
    
    ax.plot(income_range, y_non_student, 'b-', linewidth=3, 
           label=f'Non-Students (slope = {beta_income:.3f})')
    ax.plot(income_range, y_student, 'r-', linewidth=3, 
           label=f'Students (slope = {beta_income + beta_interaction:.3f})')
    
    ax.set_xlabel('Income ($1000s)', fontsize=13)
    ax.set_ylabel('Balance ($)', fontsize=13)
    ax.set_title('Income × Student Status Interaction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("INCOME × STUDENT INTERACTION")
    print("=" * 70)
    print(f"\nβ₃ (interaction) = {beta_interaction:.4f}")
    print(f"p-value = {p_interaction:.4f}")
    
    print(f"\n📊 Income effect by student status:")
    print(f"   Non-students: ${beta_income:.3f} per $1k income")
    print(f"   Students: ${beta_income + beta_interaction:.3f} per $1k income")
    print(f"   Difference: ${beta_interaction:.3f}")
    
    if p_interaction < 0.05:
        print(f"\n✅ Interaction IS significant!")
        if beta_interaction > 0:
            print(f"   Income has STRONGER effect for students")
        else:
            print(f"   Income has WEAKER effect for students")
    else:
        print(f"\n❌ Interaction NOT significant")
        print(f"   Income effect similar for both groups")
    
    print("=" * 70)


print("✅ nb3_helpers.py loaded!")
print("\n📦 Available functions:")
print("   • plot_correlation_heatmap(df)")
print("   • plot_cooks_distance(model)")
print("   • visualize_tv_radio_interaction(df, model)")
print("   • visualize_income_student_interaction(df, model)")