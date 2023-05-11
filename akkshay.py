import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats
import wbdata
import datetime
from sklearn.impute import SimpleImputer

# Load the data into dataframes
df_co2 = pd.read_csv('D:/Dataset/API_19_DS2_en_csv_v2_536159/API_19_DS2_en_csv_v2_5361599.csv', skiprows=4)
df_countries = pd.read_csv('D:/Dataset/API_19_DS2_en_csv_v2_536159/Metadata_Country_API_19_DS2_en_csv_v2_5361599.csv')
df_indicators = pd.read_csv('D:/Dataset/API_19_DS2_en_csv_v2_536159/Metadata_Indicator_API_19_DS2_en_csv_v2_5361599.csv')

def clean_data(filename):
    df = pd.read_csv(filename, skiprows=4)
    # Drop unnecessary columns
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'], axis=1)
    # Transpose dataframe
    df_t = df.set_index('Country Name').T
    # Reset index
    df_t = df_t.reset_index()
    # Rename index column
    df_t = df_t.rename(columns={'index': 'Year'})
    # Convert year column to datetime type
    df_t['Year'] = pd.to_datetime(df_t['Year'], format='%Y')
    # Set year column as index
    df_t = df_t.set_index('Year')
    # Drop rows with all NaN values
    df_t = df_t.dropna(how='all')
    # Drop columns with any NaN values
    df_t = df_t.dropna(axis=1, how='any')
    # Fill remaining NaN values with 0
    df_t = df_t.fillna(0)
    # Create dataframe with countries as columns
    df_countries = df_t.transpose()
    # Reset index
    df_countries = df_countries.reset_index()
    # Rename index column
    df_countries = df_countries.rename(columns={'index': 'Country'})
    # Create dataframe with years as columns
    df_years = df_t.reset_index()
    return df_years, df_countries


df_years, df_countries = clean_data('API_19_DS2_en_csv_v2_5361599.csv')

#Transpose df_countries
df_countries = df_countries.set_index('Country Name').transpose()

#Concatenate df_years and df_countries
df_merged = pd.concat([df_years, df_countries], axis=1)

#Reshape the data for clustering
data = df_merged.melt(id_vars=['Year'], var_name='Country Name', value_name='Value')
data.dropna(inplace=True)

# Added this function to avoid repetition
def get_wbdata(indicator, start_date, end_date):
    data = wbdata.get_data(indicator, data_date=(start_date, end_date), pandas=True)
    data = pd.DataFrame(data).reset_index()
    data.reset_index(inplace=True)
    return data

def err_ranges(x, popt, pcov):
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)
    p = len(popt)
    dof = max(0, n - p)  # degrees of freedom
    tval = scipy.stats.t.ppf(1.0 - alpha / 2.0, dof)  # student-t value for the dof and confidence level

    y_pred = exp_growth(x, *popt)

    # Calculate the Jacobian matrix
    a, b = popt
    J = np.zeros((x.size, 2))
    J[:, 0] = np.exp(b * x)
    J[:, 1] = a * x * np.exp(b * x)

    # Calculate standard errors
    sigma = np.sqrt(np.diag(np.matmul(np.matmul(J, pcov), J.T)))

    lower_bound = y_pred - tval * sigma
    upper_bound = y_pred + tval * sigma

    return lower_bound, upper_bound

# Set the time range
start_date = datetime.date(2010, 1, 1)
end_date = datetime.date(2020, 12, 31)

# Fetch data from World Bank
co2_emission = get_wbdata("EN.ATM.CO2E.PC", start_date, end_date)
gdp_data = get_wbdata("NY.GDP.PCAP.CD", start_date, end_date)
population_data = get_wbdata("SP.POP.TOTL", start_date, end_date)

# Rename columns
co2_emission.columns = ['index', 'country', 'date', 'value_co2']
gdp_data.columns = ['index', 'country', 'date', 'value_gdp']
population_data.columns = ['index', 'country', 'date', 'population']

# Merge the data
data = pd.merge(co2_emission, gdp_data, on=['country', 'date'], suffixes=('_co2', '_gdp'))
data = pd.merge(data, population_data, on=['country', 'date'])
data.rename(columns={'value': 'population'}, inplace=True)

# Calculate CO2 per $ of GDP
data['CO2_per_dollar_GDP'] = data['value_co2'] / data['value_gdp']

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[['value_co2', 'value_gdp', 'CO2_per_dollar_GDP']])

imputer = SimpleImputer(strategy='mean')
data_normalized_imputed = imputer.fit_transform(data_normalized)

# Cluster the data
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(data_normalized_imputed)

# Add cluster labels to the original data
data['cluster'] = clusters

# Plot the clusters
plt.scatter(data['value_co2'], data['value_gdp'], c=data['cluster'], cmap='viridis')
plt.xlabel('CO2 Emissions per Capita')
plt.ylabel('GDP per Capita')
plt.title('CO2 Emissions per Capita vs GDP per Capita Clustering')
plt.show()

# Define a simple model, e.g., exponential growth
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract the year from the date column
data['year'] = data['date'].dt.year

# Remove missing values
data = data.dropna()

# Fit the model
x_data = data['year']
y_data = data['value_co2']
popt, pcov = curve_fit(exp_growth, x_data, y_data, maxfev=10000)

# Make predictions and calculate confidence intervals
x_pred = np.arange(min(x_data), max(x_data) + 20, 1)

y_pred = exp_growth(x_pred, *popt)
lower_bound, upper_bound = err_ranges(x_pred, popt, pcov)

# Plot the fit and confidence intervals
plt.plot(x_data, y_data, 'k.', label='Observed CO2 Emissions')
plt.plot(x_pred, y_pred, 'r-', label='Fitted Exponential Growth')
plt.fill_between(x_pred, lower_bound, upper_bound, color='red', alpha=0.15, label='95% Confidence Interval')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions per Capita')
plt.title('CO2 Emissions per Capita Exponential Growth Model')
plt.legend()
plt.grid(True)
plt.show()

## Selecting a few countries for analysis
countries = ['United States', 'China', 'India', 'Germany']
data_subset = data[data['country'].isin(countries)]

# #Plotting the data for selected countries
for country in countries:
    country_data = data_subset[data_subset['country'] == country]
    plt.plot(country_data['year'], country_data['value_co2'], label=country)

plt.xlabel('Year')
plt.ylabel('CO2 Emissions per Capita')
plt.title('CO2 Emissions per Capita for Selected Countries')
plt.legend()
plt.grid(True)
plt.show()


# #Finding the cluster centers
cluster_centers = kmeans.cluster_centers_

# #Inverse transform the cluster centers to original scale
cluster_centers_original = scaler.inverse_transform(cluster_centers)

# Pick one country from each cluster
country_per_cluster = []
for i in range(3):  # Since we have 3 clusters
    cluster_data = data[data['cluster'] == i]
    cluster_data['distance_to_center'] = np.sqrt((cluster_data['value_co2'] - cluster_centers_original[i, 0]) ** 2 +
                                                 (cluster_data['value_gdp'] - cluster_centers_original[i, 1]) ** 2 +
                                                 (cluster_data['CO2_per_dollar_GDP'] - cluster_centers_original[i, 2]) ** 2)
    min_distance_country = cluster_data.loc[cluster_data['distance_to_center'].idxmin()]['country']
    country_per_cluster.append(min_distance_country)

# Compare the trends of the selected countries
data_subset = data[data['country'].isin(country_per_cluster)]

for country in country_per_cluster:
    country_data = data_subset[data_subset['country'] == country]
    plt.plot(country_data['year'], country_data['value_co2'], label=country)

plt.xlabel('Year')
plt.ylabel('CO2 Emissions per Capita')
plt.title('CO2 Emissions per Capita for Selected Countries (One from Each Cluster)')
plt.legend()
plt.grid(True)
plt.show()

# Set the figure size
plt.figure(figsize=(12, 8))

# Plot the clusters
plt.scatter(data['value_co2'], data['value_gdp'], c=data['cluster'], cmap='viridis', alpha=0.5)

# Plot the selected countries and label them
selected_countries_data = data[data['country'].isin(country_per_cluster)]
for i, row in selected_countries_data.iterrows():
    plt.scatter(row['value_co2'], row['value_gdp'], c='red', edgecolors='k', s=100, label=row['country'])

# Set the legend to only show unique labels
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
plt.legend(*zip(*unique_labels), loc='upper left')

plt.xlabel('CO2 Emissions per Capita')
plt.ylabel('GDP per Capita')
plt.title('CO2 Emissions per Capita vs GDP per Capita Clustering with Selected Country Labels')
plt.grid(True)
plt.show()





