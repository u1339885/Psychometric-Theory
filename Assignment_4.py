#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:38:10 2024

@author: marcwhiting
"""

import numpy as np
from scipy.stats import norm

# Example data for the problems
variance_x = 25
reliability_x = 0.8
score_x = 75

variance_y = 9
reliability_y = 0.8
score_y = 75

variance_z = 9
reliability_z = 0.6
score_z = 75

# What is the value of the standard error of measurement
def calculate_sem(variance, reliability):
    return np.sqrt(variance) * np.sqrt(1 - reliability)

# What is the 95% confidence interval
def confidence_interval(score, sem):
    z_value = norm.ppf(0.975) 
    margin_of_error = z_value * sem
    lower_bound = score - margin_of_error
    upper_bound = score + margin_of_error
    return lower_bound, upper_bound

# Function to display the results for one problem
def problem_1(variance, reliability):
    sem = calculate_sem(variance, reliability)
    print(f"Standard Error of Measurement (SEM): {sem:.2f}")
    return sem

def problem_2(score, variance, reliability):
    sem = calculate_sem(variance, reliability)
    lower_bound, upper_bound = confidence_interval(score, sem)
    print(f"95% Confidence Interval for score {score}: ({lower_bound:.2f}, {upper_bound:.2f})")
    return sem, lower_bound, upper_bound



#Problem 1
sem_x = problem_1(variance_x, reliability_x)

# Problem 2
sem_x, lower_x, upper_x = problem_2(score_x, variance_x, reliability_x)

#Problem 3
sem_y, lower_y, upper_y = problem_2(score_y, variance_y, reliability_y)


#Problem 5
sem_z, lower_z, upper_z = problem_2(score_z, variance_z, reliability_z)

# Problem 4 
print(f"\nComparison 4:")
print(f"SEM for Test X: {sem_x:.2f}, 95% CI: ({lower_x:.2f}, {upper_x:.2f})")
print(f"SEM for Test Y: {sem_y:.2f}, 95% CI: ({lower_y:.2f}, {upper_y:.2f})")


# Problem 6 
print(f"\nComparison 6:")
print(f"Reliability for Test Y: {reliability_y}, 95% CI: ({lower_y:.2f}, {upper_y:.2f})")
print(f"Reliability for Test Z: {reliability_z}, 95% CI: ({lower_z:.2f}, {upper_z:.2f})")
