#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:05:44 2024

@author: marcwhiting
"""

import pyreadstat

# Function to calculate Cronbach's Alpha
def cronbach_alpha(df):
    item_variances = df.var(axis=0, ddof=1)  # Variance of each item
    total_score_variance = df.sum(axis=1).var(ddof=1)  # Variance of total scores
    num_items = df.shape[1]  # Number of items
    alpha = (num_items / (num_items - 1)) * (1 - (item_variances.sum() / total_score_variance))
    return alpha

# Load the data
file_path = 'FilePath.SAV'
df, meta = pyreadstat.read_sav(file_path)

# Define the items for each scale
future_items = ['stpi02', 'stpi05', 'stpi06', 'stpi07', 'stpi09', 'stpi13', 'stpi15', 'stpi16', 
                'stpi18', 'stpi23', 'stpi25', 'stpi26', 'stpi28', 'stpi32', 'stpi35', 'stpi37', 
                'stpi38', 'item04f', 'item19f', 'item20f']

past_items = ['stpi03', 'stpi08', 'stpi14', 'stpi24', 'stpi31', 'item17p', 'item25p', 'item33p']

present_items = ['stpi01', 'stpi04', 'stpi10', 'stpi11', 'stpi12', 'stpi19', 'stpi20', 'stpi21', 
                 'stpi27', 'stpi29', 'stpi30', 'stpi34', 'stpi36', 'stpi22']

# Extract the relevant data for each scale
df_future = df[future_items]
df_past = df[past_items]
df_present = df[present_items]

# Calculate Cronbach's Alpha for each scale
alpha_future = cronbach_alpha(df_future)
alpha_past = cronbach_alpha(df_past)
alpha_present = cronbach_alpha(df_present)

# Print results
print("Cronbach's Alpha for each scale:")
print(f"Future scale: {alpha_future:.4f}")
print(f"Past scale: {alpha_past:.4f}")
print(f"Present scale: {alpha_present:.4f}")


# Calculate item variances for the Past scale
item_variances_past = df_past.var(axis=0, ddof=1)  # ddof=1 for sample variance

# Calculate total score variance for the Past scale
total_score_past = df_past.sum(axis=1)  # Sum the items for each respondent
total_score_variance_past = total_score_past.var(ddof=1)  # Variance of the total scores

# Number of items in the Past scale
N_past = df_past.shape[1]

# Apply Cronbach's Alpha formula
alpha_manual_past = (N_past / (N_past - 1)) * (1 - (item_variances_past.sum() / total_score_variance_past))

# Return item variances, total score variance, and manual alpha
print(item_variances_past, total_score_variance_past, alpha_manual_past)
