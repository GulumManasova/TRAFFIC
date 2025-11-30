"""
Traffic Data Analysis - Bishkek City
Exploratory Data Analysis

Student: [Your Name]
Period: August 2024 - October 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("="*80)
print("TRAFFIC DATA ANALYSIS")
print("="*80)

# Load clean data
df = pd.read_csv('clean_traffic_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = pd.to_datetime(df['date'])

print(f"\nAnalyzing {len(df):,} traffic records")
print(f"Period: {df['date'].min().strftime('%B %Y')} - {df['date'].max().strftime('%B %Y')}")

# Basic statistics
print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)

print(f"\nTravel Time:")
print(f"  Average: {df['duration_min'].mean():.1f} minutes")
print(f"  Median: {df['duration_min'].median():.1f} minutes")
print(f"  Min: {df['duration_min'].min():.1f} minutes")
print(f"  Max: {df['duration_min'].max():.1f} minutes")

print(f"\nSpeed:")
print(f"  Average: {df['avg_speed_kmh'].mean():.1f} km/h")
print(f"  Min: {df['avg_speed_kmh'].min():.1f} km/h")
print(f"  Max: {df['avg_speed_kmh'].max():.1f} km/h")

# Analysis by hour
print("\n" + "="*80)
print("ANALYSIS BY HOUR")
print("="*80)

hourly = df.groupby('hour')['duration_min'].agg(['mean', 'count'])
hourly.columns = ['avg_time', 'measurements']

print("\nTraffic patterns by hour:")
print(hourly.round(1))

print("\nSlowest hours (rush hours):")
top_hours = hourly.nlargest(5, 'avg_time')
for idx, row in top_hours.iterrows():
    print(f"  {idx:02d}:00 - {row['avg_time']:.1f} min (n={row['measurements']})")

# Plot: Hourly pattern
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(hourly.index, hourly['avg_time'], marker='o', linewidth=2.5, 
         markersize=10, color='#E74C3C', label='Average Time')
ax1.fill_between(hourly.index, hourly['avg_time'], alpha=0.3, color='#E74C3C')
ax1.axhline(y=df['duration_min'].mean(), color='blue', linestyle='--', 
            linewidth=2, alpha=0.7, label=f'Overall Avg ({df["duration_min"].mean():.1f} min)')
ax1.set_title('Average Travel Time by Hour of Day', fontsize=14, fontweight='bold')
ax1.set_xlabel('Hour', fontsize=12)
ax1.set_ylabel('Time (minutes)', fontsize=12)
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.bar(hourly.index, hourly['measurements'], color='#3498DB', alpha=0.7)
ax2.set_title('Number of Measurements by Hour', fontsize=14, fontweight='bold')
ax2.set_xlabel('Hour', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_xticks(range(0, 24, 2))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis_hourly_pattern.png', dpi=200, bbox_inches='tight')
print("\n✓ Chart saved: analysis_hourly_pattern.png")
plt.close()

# Analysis by route
print("\n" + "="*80)
print("ANALYSIS BY ROUTE")
print("="*80)

routes = df.groupby('route_name').agg({
    'duration_min': ['mean', 'min', 'max', 'count']
}).round(1)
routes.columns = ['avg', 'min', 'max', 'count']
routes = routes.sort_values('avg', ascending=False)

print("\nRoute statistics:")
print(routes)

# Plot: Routes
plt.figure(figsize=(12, 8))
routes_sorted = routes.sort_values('avg')
bars = plt.barh(range(len(routes_sorted)), routes_sorted['avg'], color='#2ECC71')
plt.yticks(range(len(routes_sorted)), routes_sorted.index)
plt.xlabel('Average Time (minutes)', fontsize=12)
plt.title('Average Travel Time by Route', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

for i, val in enumerate(routes_sorted['avg']):
    plt.text(val + 0.5, i, f'{val:.1f} min', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('analysis_routes.png', dpi=200, bbox_inches='tight')
print("✓ Chart saved: analysis_routes.png")
plt.close()

# Analysis by month
print("\n" + "="*80)
print("ANALYSIS BY MONTH (2024-2025)")
print("="*80)

# Create year-month column
df['year_month'] = df['datetime'].dt.to_period('M').astype(str)

monthly = df.groupby('year_month').agg({
    'duration_min': 'mean',
    'date': 'count'
}).round(1)
monthly.columns = ['avg_time', 'measurements']
monthly = monthly.sort_index()

print("\nMonthly averages:")
print(monthly)

# Plot: Monthly trend
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

x_pos = range(len(monthly))
ax1.plot(x_pos, monthly['avg_time'], marker='o', linewidth=2.5, 
         markersize=8, color='#9B59B6')
ax1.fill_between(x_pos, monthly['avg_time'], alpha=0.2, color='#9B59B6')
ax1.set_title('Average Travel Time by Month', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Average Time (minutes)', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(monthly.index, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

ax2.bar(x_pos, monthly['measurements'], color='#E67E22', alpha=0.7)
ax2.set_title('Data Coverage by Month', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Number of Measurements', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(monthly.index, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis_monthly_trend.png', dpi=200, bbox_inches='tight')
print("✓ Chart saved: analysis_monthly_trend.png")
plt.close()

# Seasonal analysis
print("\n" + "="*80)
print("SEASONAL ANALYSIS")
print("="*80)

seasonal = df.groupby('season')['duration_min'].agg(['mean', 'count'])
seasonal.columns = ['avg_time', 'measurements']
print("\nBy season:")
print(seasonal.round(1))

# Weekday vs Weekend
print("\n" + "="*80)
print("WEEKDAY VS WEEKEND")
print("="*80)

weekday_avg = df[df['is_weekend'] == False]['duration_min'].mean()
weekend_avg = df[df['is_weekend'] == True]['duration_min'].mean()
weekday_count = df[df['is_weekend'] == False].shape[0]
weekend_count = df[df['is_weekend'] == True].shape[0]

print(f"\nWeekday: {weekday_avg:.1f} min (n={weekday_count:,})")
print(f"Weekend: {weekend_avg:.1f} min (n={weekend_count:,})")
print(f"Difference: {weekday_avg - weekend_avg:.1f} min ({((weekday_avg/weekend_avg - 1)*100):.1f}%)")

# Plot: Weekday vs Weekend
plt.figure(figsize=(10, 6))
data = pd.DataFrame({
    'Type': ['Weekday', 'Weekend'],
    'Time': [weekday_avg, weekend_avg],
    'Count': [weekday_count, weekend_count]
})

bars = plt.bar(data['Type'], data['Time'], color=['#E74C3C', '#2ECC71'], 
               edgecolor='black', linewidth=1.5)
plt.title('Weekday vs Weekend Traffic', fontsize=14, fontweight='bold')
plt.ylabel('Average Time (minutes)', fontsize=12)
plt.ylim(0, max(data['Time']) * 1.2)

for i, (bar, count) in enumerate(zip(bars, data['Count'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f} min\n(n={count:,})', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('analysis_weekday_weekend.png', dpi=200, bbox_inches='tight')
print("✓ Chart saved: analysis_weekday_weekend.png")
plt.close()

# Data coverage visualization
print("\n" + "="*80)
print("DATA GAPS ANALYSIS")
print("="*80)

dates = df['date'].unique()
dates = pd.to_datetime(sorted(dates))

gaps = []
for i in range(1, len(dates)):
    gap_days = (dates[i] - dates[i-1]).days
    if gap_days > 1:
        gaps.append({
            'from': dates[i-1].strftime('%Y-%m-%d'),
            'to': dates[i].strftime('%Y-%m-%d'),
            'days': gap_days
        })

print(f"\nFound {len(gaps)} gaps in data collection:")
gaps_df = pd.DataFrame(gaps)
if len(gaps) > 0:
    print(gaps_df.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total records analyzed: {len(df):,}")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Days with data: {len(dates)}")
print(f"Data gaps: {len(gaps)}")
print(f"Routes: {df['route_name'].nunique()}")
print(f"Average travel time: {df['duration_min'].mean():.1f} minutes")
print(f"Charts created: 4")

print("\n✅ Analysis complete!")
