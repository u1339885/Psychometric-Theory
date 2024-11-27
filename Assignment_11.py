#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:14:33 2024

@author: marcwhiting
"""

import pandas as pd
import pyreadstat
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Filepath to the data file
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Wei/Assignment 11/MLOC.SAV'

# Load the .sav (SPSS) file into a pandas DataFrame
df, meta = pyreadstat.read_sav(file_path)

# Display basic information about the dataset
print("Dataset Information:")
df.info()

# Display the first few rows to get a sense of the data structure
print("\nFirst Few Rows of Data:")
print(df.head())

# Display column names and labels
print("\nColumn Names and Labels:")
for i in range(len(df.columns)):
    column = df.columns[i]
    label = meta.column_labels[i]
    print(f"{column}: {label}")

# Optionally, show some descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(df.describe())

# Preparing the data for factor analysis - Dropping NaN values
df_clean = df.dropna()

# Function to perform factor analysis with PAF and specified rotation
def perform_factor_analysis_paf(df, n_factors=3, rotation_method='varimax'):
    # Initialize factor analyzer with PAF method
    fa = FactorAnalyzer(n_factors=n_factors, method='principal', rotation=rotation_method)
    
    # Fit the model
    fa.fit(df)
    
    # Get the loadings
    loadings = fa.loadings_
    
    # Apply suppression of absolute values less than 0.10
    loadings[np.abs(loadings) < 0.10] = 0
    
    # Convert loadings to DataFrame for better display
    loadings_df = pd.DataFrame(loadings, index=df.columns, columns=[f'Factor {i+1}' for i in range(n_factors)])
    
    # Sort the loadings by size for each factor
    for col in loadings_df.columns:
        loadings_df[col] = loadings_df[col].abs()
    loadings_df['max_loading'] = loadings_df.max(axis=1)
    loadings_df = loadings_df.sort_values('max_loading', ascending=False)
    loadings_df.drop('max_loading', axis=1, inplace=True)
    
    print(f"\nFactor Loadings ({rotation_method.capitalize()} Rotation):")
    print(loadings_df)
    
    return loadings_df

# Perform Factor Analysis with Varimax rotation
print("\nRunning Factor Analysis with Varimax Rotation:")
rotated_loadings_varimax = perform_factor_analysis_paf(df_clean, n_factors=3, rotation_method='varimax')

# Perform Factor Analysis with Oblimin rotation
print("\nRunning Factor Analysis with Oblimin Rotation:")
rotated_loadings_oblimin = perform_factor_analysis_paf(df_clean, n_factors=3, rotation_method='oblimin')

# Function to plot Scree Plot using eigenvalues from factor analysis
def plot_scree(df):
    fa = FactorAnalyzer(n_factors=df.shape[1], method='principal')
    fa.fit(df)
    eigenvalues, _ = fa.get_eigenvalues()
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='--')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.grid()
    plt.show()
    
    # Provide eigenvalues data as output
    print("\nEigenvalues:")
    for idx, value in enumerate(eigenvalues, start=1):
        print(f"Factor {idx}: Eigenvalue = {value}")

# Plot the Scree Plot
print("\nGenerating Scree Plot:")
plot_scree(df_clean)
