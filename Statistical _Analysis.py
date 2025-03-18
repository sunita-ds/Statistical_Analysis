#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:24:56 2025

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

np.random.seed(42)
data={"product_id":range(1,21),
      "product_name":[f"product{i}" for i in range(1,21)],
      "category":np.random.choice(["home","sports","electronics","clothing"],20),
      "units_sold":np.random.poisson(lam=20,size=20),
      "sale_date":pd.date_range(start='2023-01-01',periods=20,freq='D')}
                                
sales_data=pd.DataFrame(data)
print("sales_data:")
print(sales_data)

sales_data.to_csv('sales_data.csv',index=False)

import os
os.getcwd()

descriptive_stats=sales_data['units_sold'].describe()
print(descriptive_stats)

sales_mean=sales_data['units_sold'].mean()
sales_median=sales_data['units_sold'].median()
sales_mode=sales_data['units_sold'].mode()[0]
sales_var=sales_data['units_sold'].var()
sales_std=sales_data['units_sold'].std()

print(sales_mean)
print(sales_median)
print(sales_mode)
print(sales_var)
print(sales_std)

category_stats=sales_data.groupby('category')['units_sold'].agg(['sum','mean','std']).reset_index()
category_stats.columns=['category','total units sold','avg units sold','std of units sold']
print(category_stats)
print(category_stats.columns)

confidence_level=0.95
degrees_freedom=len(sales_data['units_sold'])-1
sample_mean=sales_mean
sample_standard_error=sales_std/np.sqrt(len(sales_data['units_sold']))
t_score=stats.t.ppf((1+confidence_level)/2,degrees_freedom)
margin_of_error=t_score*sample_standard_error
confidence_interval=(sample_mean-margin_of_error,sample_mean+margin_of_error)
print(confidence_interval)


confidence_level=0.99
degrees_freedom=len(sales_data['units_sold'])-1
sample_mean=sales_mean
sample_standard_error=sales_std/np.sqrt(len(sales_data['units_sold']))
t_score=stats.t.ppf((1+confidence_level)/2,degrees_freedom)
margin_of_error=t_score*sample_standard_error
confidence_interval=(sample_mean-margin_of_error,sample_mean+margin_of_error)
print("confidence interval for the mean of units sold:")
print(confidence_interval)

t_statistics,p_value=stats.ttest_1samp(sales_data['units_sold'],20)
print("Hypothesis Testing t_test:")
print(f"T_statistics:{t_statistics},p_value:{p_value}")
if p_value<0.05:
    print("reject the null hypothesis")
else:
    print("fail to reject the null hypothesis")
    
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))
sns.histplot(sales_data['units_sold'],bins=20,kde=True)
plt.title("Distribution of units sold")
plt.xlabel('units sold')
plt.ylabel('Frequency')
plt.axvline(sales_mean,c="red",ls="--",label='Mean')
plt.axvline(sales_median,c="green",ls="--",label='Median')
plt.axvline(sales_mode,c="blue",ls="--",label='Mode')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=sales_data,x='category',y='units_sold')
plt.title("boxplot for units sold group by category")
plt.xlabel=('category')
plt.ylabel=('units sold')
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(data=category_stats,x='category',y='total units sold')
plt.title("total units sold by category")
plt.xlabel=('category')
plt.ylabel=('total units sold')
plt.show()

