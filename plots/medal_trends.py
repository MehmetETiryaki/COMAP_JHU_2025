import matplotlib.pyplot as plt
import pandas as pd

# Load the medal counts dataset
df = pd.read_csv('data/raw/summerOly_medal_counts.csv')

# Define host country mapping based on the provided list
host_countries = {
    1896: 'Greece', 1900: 'France', 1904: 'United States', 1908: 'United Kingdom',
    1912: 'Sweden', 1920: 'Belgium', 1924: 'France', 1928: 'Netherlands',
    1932: 'United States', 1936: 'Germany', 1948: 'United Kingdom', 1952: 'Finland',
    1956: 'Australia', 1960: 'Italy', 1964: 'Japan', 1968: 'Mexico',
    1972: 'West Germany', 1976: 'Canada', 1980: 'Soviet Union', 1984: 'United States',
    1988: 'South Korea', 1992: 'Spain', 1996: 'United States', 2000: 'Australia',
    2004: 'Greece', 2008: 'China', 2012: 'United Kingdom', 2016: 'Brazil',
    2020: 'Japan', 2024: 'France'
}

# Countries to analyze
countries = ['United States', 'China', 'Soviet Union', 'Russia', 'Greece']

# Plot medal trends for selected countries
for country in countries:
    country_data = df[df['NOC'] == country]

    # Plot total medals for each country
    plt.plot(country_data['Year'], country_data['Total'], label=f'Total for {country}')

    # Find host years for this country
    host_years = [year for year, host in host_countries.items() if host == country]
    host_data = country_data[country_data['Year'].isin(host_years)]

    # Highlight home country years with red scatter points
    plt.scatter(host_data['Year'], host_data['Total'], color='red', label='_nolegend_')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Total Medals')
plt.title('Olympic Medal Trends with Home Advantage')
plt.legend()

plt.grid(True)
plt.show()