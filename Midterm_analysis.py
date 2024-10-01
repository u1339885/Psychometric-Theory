#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:50:29 2024

@author: marcwhiting
"""

import pyreadstat
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import os

# Load the 'DPS_All.sav' file
folder_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Wei/Midterm/sav Files'
file_path = os.path.join(folder_path, 'DPS_All.sav')
df_all, meta = pyreadstat.read_sav(file_path)

# Step 1: Calculate Cronbach's Alpha for Odd and Even split-halves
odd_items = df_all['Odd']
even_items = df_all['Even']

# Combine them into a DataFrame for alpha calculation
split_data = pd.DataFrame({'Odd': odd_items, 'Even': even_items})

# Using pingouin to calculate Cronbach's Alpha for Odd/Even items
cronbach_split = pg.cronbach_alpha(split_data)[0]
print(f"Cronbach's Alpha for Split-Halves (Odd vs Even): {cronbach_split:.3f}")

# Step 2: Calculate Pearson correlation between Odd and Even
corr_forms, p_value_forms = pearsonr(odd_items, even_items)
print(f"Pearson Correlation between Odd and Even: {corr_forms:.3f}")

# Spearman-Brown Coefficient (Equal Length)
spearman_brown_eq = 2 * corr_forms / (1 + corr_forms)
print(f"Spearman-Brown Coefficient (Equal Length): {spearman_brown_eq:.3f}")

# Guttman Split-Half Coefficient
guttman_split_half = spearman_brown_eq  # Same formula
print(f"Guttman Split-Half Coefficient: {guttman_split_half:.3f}")

# Step 3: Calculate Cronbach's Alpha for the 54 items in DPS1 and DPS2
# Load both datasets
data_path_dps1 = os.path.join(folder_path, 'DPS1.sav')
data_path_dps2 = os.path.join(folder_path, 'DPS2.sav')

df_dps1, meta_dps1 = pyreadstat.read_sav(data_path_dps1)
df_dps2, meta_dps2 = pyreadstat.read_sav(data_path_dps2)

# Select only the 54 items (Atr1 to Atr54)
items_columns = [f'Atr{i}' for i in range(1, 55)]
df_dps1_items = df_dps1[items_columns]
df_dps2_items = df_dps2[items_columns]

# Function to calculate Cronbach's Alpha
def cronbach_alpha(data):
    items = len(data.columns)
    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)
    alpha = (items / (items - 1)) * (1 - (item_variances.sum() / total_variance))
    return alpha

# Calculate Cronbach's Alpha for DPS1 (occasion 1)
cronbach_alpha_dps1 = cronbach_alpha(df_dps1_items)
print(f"Cronbach's Alpha for DPS1 (occasion 1) 54 items: {cronbach_alpha_dps1:.3f}")

# Calculate Cronbach's Alpha for DPS2 (occasion 2)
cronbach_alpha_dps2 = cronbach_alpha(df_dps2_items)
print(f"Cronbach's Alpha for DPS2 (occasion 2) 54 items: {cronbach_alpha_dps2:.3f}")
