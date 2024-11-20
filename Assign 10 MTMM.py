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
            method_row, trait_row = row.split('_')
            method_col, trait_col = col.split('_')
            if row == col:
                # Reliability values, ignore in MTMM context
                labeled_matrix.loc[row, col] = f"Reliability ({corr_matrix.loc[row, col]:.2f})"
            elif method_row != method_col and trait_row == trait_col:
                # Monotrait-Heteromethod (MH)
                labeled_matrix.loc[row, col] = f"MH ({corr_matrix.loc[row, col]:.2f})"
            elif method_row == method_col and trait_row != trait_col:
                # Heterotrait-Monomethod (HM)
                labeled_matrix.loc[row, col] = f"HM ({corr_matrix.loc[row, col]:.2f})"
            else:
                # Heterotrait-Heteromethod (HH)
                labeled_matrix.loc[row, col] = f"HH ({corr_matrix.loc[row, col]:.2f})"
    return labeled_matrix

# Labeling the correlation matrix
labeled_correlation_matrix = label_mtmm_coefficients(correlation_matrix)
print("\nLabeled Multitrait-Multimethod Correlation Matrix:")
print(labeled_correlation_matrix)

# Save the labeled correlation matrix to a CSV for reference
labeled_correlation_matrix.to_csv("MTMM_Labeled_Correlation_Matrix.csv", index=True)

# Reliability coefficients for each scale
reliabilities = {
    'STPI_Future': 0.77,
    'STPI_Past': 0.56,
    'STPI_Present': 0.69,
    'BTOI_Future': 0.77,
    'BTOI_Past': 0.79,
    'BTOI_Present': 0.75
}

# Calculate disattenuated correlations for monotrait-heteromethod
def calculate_disattenuated_correlations(corr_matrix, reliabilities):
    disattenuated_corrs = {}
    for row in corr_matrix.index:
        for col in corr_matrix.columns:
            method_row, trait_row = row.split('_')
            method_col, trait_col = col.split('_')
            if method_row != method_col and trait_row == trait_col:  # Monotrait-Heteromethod
                reliability_x = reliabilities.get(row, 1)
                reliability_y = reliabilities.get(col, 1)
                observed_corr = corr_matrix.loc[row, col]
                disattenuated_corr = observed_corr / (np.sqrt(reliability_x * reliability_y))
                disattenuated_corrs[(row, col)] = disattenuated_corr
                print(f"Disattenuated correlation between {row} and {col}: {disattenuated_corr:.2f}")
    return disattenuated_corrs

# Calculate and print disattenuated correlations
print("\nDisattenuated Monotrait-Heteromethod Correlations:")
disattenuated_corrs = calculate_disattenuated_correlations(correlation_matrix, reliabilities)

# Extract specific correlation types and output them for interpretation
mh_correlations = []
hh_correlations = []
hm_correlations = []
for row in labeled_correlation_matrix.index:
    for col in labeled_correlation_matrix.columns:
        value = labeled_correlation_matrix.loc[row, col]
        if "MH" in value:
            mh_correlations.append((row, col, correlation_matrix.loc[row, col]))
        elif "HH" in value:
            hh_correlations.append((row, col, correlation_matrix.loc[row, col]))
        elif "HM" in value:
            hm_correlations.append((row, col, correlation_matrix.loc[row, col]))

# Output the numerical values for MH, HH, HM correlations for further interpretation
print("\nMonotrait-Heteromethod (MH) Correlations:")
for item in mh_correlations:
    print(f"{item[0]} vs {item[1]}: {item[2]:.2f}")

print("\nHeterotrait-Heteromethod (HH) Correlations:")
for item in hh_correlations:
    print(f"{item[0]} vs {item[1]}: {item[2]:.2f}")

print("\nHeterotrait-Monomethod (HM) Correlations:")
for item in hm_correlations:
    print(f"{item[0]} vs {item[1]}: {item[2]:.2f}")

# Function to convert labeled matrix to numeric matrix for plotting
# and to create a corresponding annotation matrix with MH, HH, HM labels
def extract_numeric_and_labels(labeled_matrix):
    numeric_matrix = pd.DataFrame(index=labeled_matrix.index, columns=labeled_matrix.columns, dtype=float)
    labels_matrix = pd.DataFrame(index=labeled_matrix.index, columns=labeled_matrix.columns, dtype=object)
    for row in labeled_matrix.index:
        for col in labeled_matrix.columns:
            value = labeled_matrix.loc[row, col]
            if "Reliability" in value:
                numeric_matrix.loc[row, col] = 1.0
                labels_matrix.loc[row, col] = "Reliability"
            else:
                corr_value = float(value.split("(")[1].strip(")"))
                numeric_matrix.loc[row, col] = corr_value
                if "MH" in value:
                    labels_matrix.loc[row, col] = "MH"
                elif "HM" in value:
                    labels_matrix.loc[row, col] = "HM"
                elif "HH" in value:
                    labels_matrix.loc[row, col] = "HH"
    return numeric_matrix, labels_matrix

# Extract numeric matrix and labels matrix for annotations
numeric_matrix, labels_matrix = extract_numeric_and_labels(labeled_correlation_matrix)

# Create a heatmap with annotations to label MH, HH, HM
plt.figure(figsize=(10, 8))
ax = sns.heatmap(numeric_matrix, annot=True, cmap='coolwarm', linewidths=0.5,
                 cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f")
# Modify the annotate_heatmap function
def annotate_heatmap(ax, labels_matrix):
    for i in range(labels_matrix.shape[0]):
        for j in range(labels_matrix.shape[1]):
            label = labels_matrix.iloc[i, j]
            if label != "Reliability":
                # Place the label at the top of the cell
                ax.text(j + 0.5, i + 0.35, label, ha='center', va='center', color='black', fontsize=8)
            else:
                # For reliability, you can choose to display it differently or not at all
                pass

# Adjust the heatmap call to remove numerical annotations
plt.figure(figsize=(10, 8))
ax = sns.heatmap(numeric_matrix, cmap='coolwarm', linewidths=0.5,
                 cbar_kws={'label': 'Correlation Coefficient'}, annot=True, fmt=".2f", annot_kws={"size": 8})

annotate_heatmap(ax, labels_matrix)



plt.title("Multitrait-Multimethod Correlation Matrix with Validity Coefficients (MH, HH, HM)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




