#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:17:40 2024

@author: marcwhiting
"""


import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, norm
import numpy as np

# Load the datasets using pyreadstat
procspd_df, procspd_meta = pyreadstat.read_sav("FileName")
stpi_df, stpi_meta = pyreadstat.read_sav("/FileName")

# Analyze means and standard deviations
def analyze_means_std(df, columns):
    means = df[columns].mean()
    stds = df[columns].std()
    return means, stds

# Calculate Pearson correlation coefficients (test-retest reliability)
def calculate_reliability(df, columns):
    correlation, _ = pearsonr(df[columns[0]], df[columns[1]])
    return correlation

# Column pairs for the two occasions for each measure
columns_pairs = {
    'CRT': ['CRT1', 'CRT2'],
    'CATID': ['CATID1', 'CATID2'],
    'SEMREL': ['SEMREL1', 'SEMREL2'],
    'SEMID': ['SEMID1', 'SEMID2'],
    'SEMREC': ['SEMREC1', 'SEMREC2']
}

# Calculate means, standard deviations, and reliability coefficients
results = {}
for key, cols in columns_pairs.items():
    means, stds = analyze_means_std(procspd_df, cols)
    reliability = calculate_reliability(procspd_df, cols)
    results[key] = {'Means': means, 'Standard Deviations': stds, 'Reliability': reliability}

# Display results
for key, value in results.items():
    print(f"{key} - Means: {value['Means']}, Standard Deviations: {value['Standard Deviations']}, Reliability: {value['Reliability']:.3f}")

# Plot univariate histograms with normal curve
for key, cols in columns_pairs.items():
    df_subset = procspd_df[cols]
    for col in cols:
        plt.figure()
        sns.histplot(df_subset[col], kde=False, stat="density", bins=10)
        mn, std = df_subset[col].mean(), df_subset[col].std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mn, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f' {col}')
        plt.show()

# Plot bivariate scattergrams
for key, cols in columns_pairs.items():
    sns.scatterplot(x=procspd_df[cols[0]], y=procspd_df[cols[1]])
    plt.title(f'Scatterplot for {key} - {cols[0]} vs {cols[1]}')
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()
    
