#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data emissions data from 1990-2050 with Virginia and West Virginia removed (talk to Tim about that)
df = pd.read_csv('cleaned_co2_data_non_va_wv.csv')


# In[3]:


#Find the top 10 counties in U.S. by emissions in 2005 from the dataset
#Chatgpt is good for identifying counties by FIPS code
top_10_2005 = df[df['year'] == 2005].nlargest(10, 'emissions')[['geoid', 'emissions']]
print(top_10_2005)


# In[5]:


# Find the top 10 counties by emissions in New York State for 2005(FIPS code starts with 36)
# again chatgpt is good for identifying which counties are which
ny_2005_emissions = df[(df['year'] == 2005) & (df['geoid'].astype(str).str.startswith('36'))]
top_10_ny = ny_2005_emissions[ny_2005_emissions['year'] == 2005].nlargest(10,'emissions')[['geoid', 'emissions']]
print(top_10_ny)


# In[8]:


#plot emissions trends assuming that tasks are taken on starting in 2020
co2_data_full = df.copy()

top_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
top_counties = {count: co2_data_full[co2_data_full['year'] == 2005].nlargest(count, 'emissions')['geoid'].unique()
                for count in top_counts}

# Identifying the top 100, 200, ..., 1000 counties in 2005
top_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
top_counties = {count: co2_data_full[co2_data_full['year'] == 2005].nlargest(count, 'emissions')['geoid'].unique()
                for count in top_counts}

# Prepare data for reduction from 2020 onwards
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
reductions = {2025: 0.1294, 2030: 0.1294, 2035: 0.2831, 2040: 0.2831, 2045: 0.2831, 2050: 0.2831}

# Function to apply reduction
def apply_scenario_reduction(data, top_counties, reduction_years):
    data = data.copy()
    for year in reduction_years:
        if year > 2020:
            previous_year = year - 5
            rate = 1 - reductions[year]
            data.loc[(data['geoid'].isin(top_counties)) & (data['year'] == year), 'emissions'] *= rate
            # Apply recursively for following years
            for future_year in range(year+5, 2051, 5):
                data.loc[(data['geoid'].isin(top_counties)) & (data['year'] == future_year), 'emissions'] *= rate
    return data

# Applying the scenario reductions
scenario_data = {}
for count in top_counts:
    scenario_data[count] = apply_scenario_reduction(co2_data_full, top_counties[count], years)

# Calculate aggregate emissions by year for each scenario
scenario_aggregates = {count: scenario_data[count].groupby('year')['emissions'].sum() for count in top_counts}

# Also calculate the business as usual case (no reduction)
ba_aggregate = co2_data_full.groupby('year')['emissions'].sum()

# Calculate the aggregate emissions for 2005 and 50% of that
emissions_2005 = co2_data_full[co2_data_full['year'] == 2005]['emissions'].sum()
half_emissions_2005 = 0.5 * emissions_2005

# Plotting with additional horizontal lines
plt.figure(figsize=(14, 8))
plt.plot(ba_aggregate.index, ba_aggregate, label='Business as Usual', marker='o', linestyle='-')

for count in top_counts:
    plt.plot(scenario_aggregates[count].index, scenario_aggregates[count], label=f'Top {count}', marker='.')

# Adding horizontal lines
plt.axhline(y=emissions_2005, color='r', linestyle='--', label='2005 Emissions')
plt.axhline(y=half_emissions_2005, color='g', linestyle='--', label='50% of 2005 Emissions')

plt.title('Aggregate CO2 Emissions by Scenario with Baseline Markers')
plt.xlabel('Year')
plt.ylabel('Emissions (kg)')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


#plot emissions trends assuming that tasks were taken on starting in 2005
co2_data_full = df.copy()
# Prepare data for reduction from 2005 onwards
years_2005 = [2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
reductions_2005 = {2005: .0602, 2010: .0602, 2015: .0602, 2020: .0602, 2025: 0.1294, 2030: 0.1294, 2035: 0.2831, 2040: 0.2831, 2045: 0.2831, 2050: 0.2831}

# Function to apply reduction
def apply_scenario_reduction_2005(data, top_counties, reduction_years):
    data = data.copy()
    for year in reduction_years:
        if year > 2005:
            previous_year = year - 5
            rate = 1 - reductions_2005[year]
            data.loc[(data['geoid'].isin(top_counties)) & (data['year'] == year), 'emissions'] *= rate
            # Apply recursively for following years
            for future_year in range(year+5, 2051, 5):
                data.loc[(data['geoid'].isin(top_counties)) & (data['year'] == future_year), 'emissions'] *= rate
    return data

# Applying the scenario reductions
scenario_2005_data = {}
for count in top_counts:
    scenario_2005_data[count] = apply_scenario_reduction_2005(co2_data_full, top_counties[count], years_2005)

# Calculate aggregate emissions by year for each scenario
scenario_2005_aggregates = {count: scenario_2005_data[count].groupby('year')['emissions'].sum() for count in top_counts}

# Also calculate the business as usual case (no reduction)
ba_aggregate = co2_data_full.groupby('year')['emissions'].sum()

# Calculate the aggregate emissions for 2005 and 50% of that
emissions_2005 = co2_data_full[co2_data_full['year'] == 2005]['emissions'].sum()
half_emissions_2005 = 0.5 * emissions_2005

# Plotting with additional horizontal lines
plt.figure(figsize=(14, 8))
plt.plot(ba_aggregate.index, ba_aggregate, label='Business as Usual', marker='o', linestyle='-')

for count in top_counts:
    plt.plot(scenario_2005_aggregates[count].index, scenario_2005_aggregates[count], label=f'Top {count}', marker='.')

# Adding horizontal lines
plt.axhline(y=emissions_2005, color='r', linestyle='--', label='2005 Emissions')
plt.axhline(y=half_emissions_2005, color='g', linestyle='--', label='50% of 2005 Emissions')

plt.title('Aggregate CO2 Emissions by Scenario with Baseline Markers')
plt.xlabel('Year')
plt.ylabel('Emissions (kg)')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


emissions_2005 = df[df['year'] == 2005]['emissions']
plt.figure(figsize=(10, 6))
plt.hist(emissions_2005, bins=1000, color='skyblue', edgecolor='black')
plt.title('Histogram of Emissions in 2005')
plt.xlabel('Emissions')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[22]:


from scipy.stats import lognorm

# Fit a log-normal distribution to the emissions data for 2005
shape, loc, scale = lognorm.fit(emissions_2005)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(emissions_2005, bins=1000, density=True, color='skyblue', edgecolor='skyblue', alpha=0.6)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = lognorm.pdf(x, shape, loc, scale)
plt.plot(x, p, 'k', linewidth=2)
title = f"Fit results: shape = {shape:.2f}, loc = {loc:.2f}, scale = {scale:.2f}"
plt.title(title)
plt.xlabel('Emissions')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:




