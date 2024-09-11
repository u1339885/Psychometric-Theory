#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:44:46 2024

@author: marcwhiting
"""

import pyreadstat
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# Load the data
file_path = 'FilePath.SAV'
df, meta = pyreadstat.read_sav(file_path)

# List of odd and even variables for each scale
future_even = 'stpifue'
future_odd = 'stpifuo'
past_even = 'stpipae'
past_odd = 'stpipao'
present_even = 'stpipre'
present_odd = 'stpipro'

# Function to create histograms with normal curve
def plot_histogram(df, col_even, col_odd, scale_name):
    plt.figure(figsize=(14, 6))

    # Histogram for even items
    plt.subplot(1, 2, 1)
    sns.histplot(df[col_even], kde=True, stat='density')
    plt.title(f"{scale_name} Even Items")

    # Histogram for odd items
    plt.subplot(1, 2, 2)
    sns.histplot(df[col_odd], kde=True, stat='density')
    plt.title(f"{scale_name} Odd Items")

    plt.tight_layout()
    plt.show()

# Scatterplot with a line of best fit
def plot_scatter_and_correlation(df, col_even, col_odd, scale_name):
    plt.figure(figsize=(6, 6))

    # Scatterplot with regression line
    sns.regplot(x=df[col_odd], y=df[col_even], scatter_kws={'s':50})
    plt.title(f"Scatterplot for {scale_name} Odd vs Even Items")
    plt.xlabel(f"{scale_name} Odd Items")
    plt.ylabel(f"{scale_name} Even Items")
    plt.show()

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(df[col_odd], df[col_even])
    
    # Check if p-value is less than 0.05
    significance = "< 0.05" if p_value < 0.05 else ">= 0.05"
    print(f"Pearson correlation for {scale_name}: {correlation:.4f} (p-value: {significance})")
    return correlation

# Spearman-Brown formula for split-half reliability
def spearman_brown(correlation):
    return (2 * correlation) / (1 + correlation)

# Future scale: Histograms, scatterplot, and correlation
print("=== Future Scale ===")
plot_histogram(df, future_even, future_odd, "Future")
future_corr = plot_scatter_and_correlation(df, future_even, future_odd, "Future")

# Past scale: Histograms, scatterplot, and correlation
print("\n=== Past Scale ===")
plot_histogram(df, past_even, past_odd, "Past")
past_corr = plot_scatter_and_correlation(df, past_even, past_odd, "Past")

# Present scale: Histograms, scatterplot, and correlation
print("\n=== Present Scale ===")
plot_histogram(df, present_even, present_odd, "Present")
present_corr = plot_scatter_and_correlation(df, present_even, present_odd, "Present")

# Calculate split-half reliabilities
future_reliability = spearman_brown(future_corr)
past_reliability = spearman_brown(past_corr)
present_reliability = spearman_brown(present_corr)

# Display split-half reliabilities
print("\nSplit-half Reliability Estimates:")
print(f"Future scale reliability: {future_reliability:.4f}")
print(f"Past scale reliability: {past_reliability:.4f}")
print(f"Present scale reliability: {present_reliability:.4f}")
