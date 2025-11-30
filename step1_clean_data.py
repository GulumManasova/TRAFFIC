"""
Traffic Data Analysis - Bishkek City
Data Cleaning and Preparation

Student: [Your Name]
Period: August 2024 - October 2025
"""

import pandas as pd
import numpy as np

print("="*80)
print("TRAFFIC DATA CLEANING")
print("="*80)

# Load data files
print("\nLoading data files...")
df_2024 = pd.read_csv('traffic_data_2024.csv')
df_2025 = pd.read_csv('traffic_data_2025.csv')

print(f"2024 data: {len(df_2024):,} rows")
print(f"2025 data: {len(df_2025):,} rows")

# Combine datasets
df = pd.concat([df_2024, df_2025], ignore_index=True)
print(f"\nTotal rows: {len(df):,}")

# Check date range
print(f"Period: {df['date'].min()} to {df['date'].max()}")
print(f"Unique dates: {df['date'].nunique()}")

# Basic info
print("\n" + "="*80)
print("DATASET INFO")
print("="*80)
print(df.info())

print("\nFirst rows:")
print(df.head(10))

# Check data quality
print("\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80)

# Missing values
print("\n1. Missing values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✓ No missing values")
else:
    print(missing[missing > 0])

# Duplicates
print("\n2. Duplicate rows:")
duplicates = df.duplicated().sum()
print(f"   Found: {duplicates}")
if duplicates > 0:
    print("   Removing duplicates...")
    df = df.drop_duplicates()
    print(f"   After removal: {len(df):,} rows")

# Check for unrealistic values
print("\n3. Value ranges:")
print(f"   Duration: {df['duration_min'].min():.1f} - {df['duration_min'].max():.1f} min")
print(f"   Speed: {df['avg_speed_kmh'].min():.1f} - {df['avg_speed_kmh'].max():.1f} km/h")
print(f"   Distance: {df['distance_m'].min()} - {df['distance_m'].max()} m")

# Check for outliers
negative_time = (df['duration_min'] <= 0).sum()
negative_speed = (df['avg_speed_kmh'] <= 0).sum()
print(f"\n4. Data issues:")
print(f"   Negative duration: {negative_time}")
print(f"   Negative speed: {negative_speed}")

# Convert dates
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = pd.to_datetime(df['date'])

# Add useful features
print("\n" + "="*80)
print("CREATING NEW FEATURES")
print("="*80)

# Time period
def time_period(hour):
    if hour < 6:
        return 'Night'
    elif hour < 12:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df['time_period'] = df['hour'].apply(time_period)
print("✓ Created: time_period")

# Rush hour
df['rush_hour'] = df['hour'].isin([8, 9, 18, 19])
print("✓ Created: rush_hour")

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)
print("✓ Created: season")

# Month name
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
df['month_name'] = df['month'].map(month_names)
print("✓ Created: month_name")

# Speed category
def speed_category(speed):
    if speed < 15:
        return 'Very Slow'
    elif speed < 25:
        return 'Slow'
    elif speed < 35:
        return 'Normal'
    else:
        return 'Fast'

df['speed_cat'] = df['avg_speed_kmh'].apply(speed_category)
print("✓ Created: speed_cat")

# Statistics by month
print("\n" + "="*80)
print("DATA COVERAGE BY MONTH")
print("="*80)
monthly = df.groupby(['year', 'month_name']).size().reset_index(name='count')
print(monthly.to_string(index=False))

# Statistics by route
print("\n" + "="*80)
print("DATA COVERAGE BY ROUTE")
print("="*80)
routes = df.groupby('route_name').size().sort_values(ascending=False)
print(routes)

# Save clean data
print("\n" + "="*80)
print("SAVING CLEAN DATA")
print("="*80)

df.to_csv('clean_traffic_data.csv', index=False)
print("✓ Saved: clean_traffic_data.csv")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total records: {len(df):,}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Number of days with data: {df['date'].nunique()}")
print(f"Routes analyzed: {df['route_name'].nunique()}")
print(f"Final columns: {len(df.columns)}")

print("\n✅ Data cleaning complete!")
