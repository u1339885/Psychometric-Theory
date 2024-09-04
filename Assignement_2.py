#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:09:30 2024

@author: marcwhiting
"""

import pandas as pd
import pyreadstat

# Paths to the .sav files
procspd_path = "FileName"
stpi_path = "FileName"

# Reading the PROCSPD.SAV file
procspd_df, procspd_meta = pyreadstat.read_sav(procspd_path)

# Reading the STPI.SAV file
stpi_df, stpi_meta = pyreadstat.read_sav(stpi_path)

# Display the first few rows of the PROCSPD dataset
print(procspd_df.head())

# Display the first few rows of the STPI dataset
print(stpi_df.head())

# Get information about the columns in the PROCSPD dataset
print(procspd_df.info())

# Get information about the columns in the STPI dataset
print(stpi_df.info())

# Optionally, display the metadata to understand variable labels
print(procspd_meta.column_labels)
print(stpi_meta.column_labels)

# Descriptive statistics for PROCSPD dataset
print(procspd_df.describe())

# Descriptive statistics for STPI dataset
print(stpi_df.describe())
