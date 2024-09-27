#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:04:23 2024

@author: marcwhiting
"""

import pandas as pd
import pyreadstat
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Wei/Assignment 5/STPI.SAV'
df, meta = pyreadstat.read_sav(file_path)

# Define the items for each scale
future_items = ['stpi02', 'stpi05', 'stpi06', 'stpi07', 'stpi09', 'stpi13', 'stpi15', 'stpi16', 
                'stpi18', 'stpi23', 'stpi25', 'stpi26', 'stpi28', 'stpi32', 'stpi35', 'stpi37', 
                'stpi38', 'item04f', 'item19f', 'item20f']

past_items = ['stpi03', 'stpi08', 'stpi14', 'stpi24', 'stpi31', 'item17p', 'item25p', 'item33p']

present_items = ['stpi01', 'stpi04', 'stpi10', 'stpi11', 'stpi12', 'stpi19', 'stpi20', 'stpi21', 
                 'stpi27', 'stpi29', 'stpi30', 'stpi34', 'stpi36', 'stpi22']

# Extract data for each scale
df_scales = {
    "Future": df[future_items],
    "Past": df[past_items],
    "Present": df[present_items]
}

# Function to compute variance components using two-way ANOVA
def compute_variance_components(df, n_items, n_people):
    df_long = pd.melt(df.reset_index(), id_vars="index", var_name="Item", value_name="Score")
    df_long = df_long.rename(columns={"index": "Person"})
    
    model = ols('Score ~ C(Person) + C(Item)', data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    MS_between_people = anova_table.loc["C(Person)", "sum_sq"] / anova_table.loc["C(Person)", "df"]
    MS_between_items = anova_table.loc["C(Item)", "sum_sq"] / anova_table.loc["C(Item)", "df"]
    MS_residual = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]

    var_people = (MS_between_people - MS_residual) / n_items
    var_items = (MS_between_items - MS_residual) / n_people
    var_residual = MS_residual
    
    return var_people, var_items, var_residual

# Generalizability coefficient (G) for relative decisions
def compute_generalizability_relative(var_people, var_residual, n_items):
    return var_people / (var_people + (var_residual / n_items))

# Generalizability coefficient (Φ) for absolute decisions
def compute_generalizability_absolute(var_people, var_items, var_residual, n_items, n_people):
    return var_people / (var_people + (var_residual / n_items) + (var_items / n_people))

# Function to compute both G and Φ for 8 and 20 items
def compute_generalizability(var_people, var_items, var_residual, n_people):
    results = {}
    for n_items in [8, 20]:
        G = compute_generalizability_relative(var_people, var_residual, n_items)
        Phi = compute_generalizability_absolute(var_people, var_items, var_residual, n_items, n_people)
        results[n_items] = {'G': G, 'Phi': Phi}
    return results

# Compute and display generalizability results for each scale
for scale_name, df_scale in df_scales.items():
    n_items_scale = df_scale.shape[1]
    n_people_scale = df_scale.shape[0]
    
    # Compute variance components
    var_people, var_items, var_residual = compute_variance_components(df_scale, n_items_scale, n_people_scale)
    
    # Compute generalizability coefficients for 8 and 20 items
    generalizability_results = compute_generalizability(var_people, var_items, var_residual, n_people_scale)
    
    print(f"\nGeneralizability Coefficients for {scale_name} Scale:\n")
    
    # Display variance components
    print(f"Person Variance: {var_people:.4f}, Item Variance: {var_items:.4f}, Residual Variance: {var_residual:.4f}")
    
    # Display results for 8 and 20 items
    for n_items, result in generalizability_results.items():
        print(f"{n_items} items -> G: {result['G']:.4f}, Φ: {result['Phi']:.4f}")

