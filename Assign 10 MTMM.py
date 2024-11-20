# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import pyreadstat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the .sav file and preview the contents
filepath = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Wei/Assignment 10/MTMM.SAV'

# Read the SPSS file using pyreadstat
data, meta = pyreadstat.read_sav(filepath)

# Rename columns for easier understanding
data.rename(columns={
    'btoi_fu': 'BTOI_Future',
    'btoi_pa': 'BTOI_Past',
    'btoi_pr': 'BTOI_Present',
    'stpi_fu': 'STPI_Future',
    'stpi_pa': 'STPI_Past',
    'stpi_pr': 'STPI_Present'
}, inplace=True)

# Display the first few rows of the dataframe to understand the structure
print("First 5 rows of the dataset:")
print(data.head())

# Display metadata to understand variable labels and value labels
print("\nMetadata (variable labels):")
for variable in meta.column_labels:
    print(variable)

# Save the data as a CSV for easier exploration if needed
data.to_csv("MTMM_Exploration.csv", index=False)

# Alternatively, to visually see it in a DataFrame format in Spyder IDE
try:
    from spyder_utility import display_dataframe_to_user
    display_dataframe_to_user("MTMM Data Exploration", data)
except ImportError:
    print("Spyder utility not available. Please view the CSV or use another method to explore the data.")

# Calculate the multitrait-multimethod (MTMM) correlation matrix
correlation_matrix = data.corr()
print("\nMultitrait-Multimethod Correlation Matrix:")
print(correlation_matrix)

# Function to label types of validity coefficients in the MTMM matrix
def label_mtmm_coefficients(corr_matrix):
    labeled_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns, dtype=object)
    for row in corr_matrix.index:
        for col in corr_matrix.columns:
            if row == col:
                # Reliability values, ignore in MTMM context
                labeled_matrix.loc[row, col] = f"Reliability ({corr_matrix.loc[row, col]:.2f})"
            elif row[:4] == col[:4] and row != col:
                # Monotrait-Heteromethod (MH)
                labeled_matrix.loc[row, col] = f"MH ({corr_matrix.loc[row, col]:.2f})"
            elif row[:4] != col[:4] and row[5:] == col[5:]:
                # Heterotrait-Monomethod (HM)
                labeled_matrix.loc[row, col] = f"HM ({corr_matrix.loc[row, col]:.2f})"
            elif row[:4] != col[:4] and row[5:] != col[5:]:
                # Heterotrait-Heteromethod (HH)
                labeled_matrix.loc[row, col] = f"HH ({corr_matrix.loc[row, col]:.2f})"
    return labeled_matrix

# Labeling the correlation matrix
labeled_correlation_matrix = label_mtmm_coefficients(correlation_matrix)
print("\nLabeled Multitrait-Multimethod Correlation Matrix:")
print(labeled_correlation_matrix)

# Visualize the original correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Multitrait-Multimethod Correlation Matrix")
plt.show()

# Save the labeled correlation matrix to a CSV for reference
labeled_correlation_matrix.to_csv("MTMM_Labeled_Correlation_Matrix.csv", index=True)

# Reliability coefficients for each scale
reliabilities = {
    'STPI_FU': 0.77,
    'STPI_PA': 0.56,
    'STPI_PR': 0.69,
    'BTOI_FU': 0.77,
    'BTOI_PA': 0.79,
    'BTOI_PR': 0.75
}

# Calculate disattenuated correlations for monotrait-heteromethod
def calculate_disattenuated_correlations(corr_matrix, reliabilities):
    disattenuated_corrs = {}
    for row in corr_matrix.index:
        for col in corr_matrix.columns:
            if row[:4] == col[:4] and row != col:  # Monotrait-Heteromethod
                reliability_x = reliabilities.get(row[:6].upper(), 1)
                reliability_y = reliabilities.get(col[:6].upper(), 1)
                observed_corr = corr_matrix.loc[row, col]
                disattenuated_corr = observed_corr / (np.sqrt(reliability_x * reliability_y))
                disattenuated_corrs[(row, col)] = disattenuated_corr
                print(f"Disattenuated correlation between {row} and {col}: {disattenuated_corr:.2f}")
    return disattenuated_corrs

# Calculate and print disattenuated correlations
print("\nDisattenuated Monotrait-Heteromethod Correlations:")
disattenuated_corrs = calculate_disattenuated_correlations(correlation_matrix, reliabilities)
