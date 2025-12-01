"""
Traffic Analysis - Bishkek
Step 3: Machine Learning & Clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

print("Step 3: Advanced Analysis")
print("="*60)

df = pd.read_csv('clean_traffic_data.csv')
print(f"Loaded {len(df):,} rows")

# Prepare features
encoder = LabelEncoder()
df['route_code'] = encoder.fit_transform(df['route_name'])

df['night'] = (df['hour'] < 6).astype(int)
df['morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
df['afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
df['evening'] = (df['hour'] >= 18).astype(int)
df['weekend'] = df['is_weekend'].astype(int)
df['rush'] = df['rush_hour'].astype(int)

features = ['hour', 'day_of_week', 'month', 'distance_m', 'weekend', 
            'rush', 'route_code', 'night', 'morning', 'afternoon', 'evening']

X = df[features]
y = df['duration_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# PART 1: REGRESSION MODELS (Predict time)
# ============================================================================
print("\n[Part 1] Testing Regression Models...")

results = {}

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results['Linear'] = {
    'R²': r2_score(y_test, pred_lr),
    'MAE': mean_absolute_error(y_test, pred_lr)
}

# Model 2: Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
results['Ridge'] = {
    'R²': r2_score(y_test, pred_ridge),
    'MAE': mean_absolute_error(y_test, pred_ridge)
}

# Model 3: Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)
results['Lasso'] = {
    'R²': r2_score(y_test, pred_lasso),
    'MAE': mean_absolute_error(y_test, pred_lasso)
}

# Model 4: Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'R²': r2_score(y_test, pred_dt),
    'MAE': mean_absolute_error(y_test, pred_dt)
}

# Model 5: Random Forest
rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'R²': r2_score(y_test, pred_rf),
    'MAE': mean_absolute_error(y_test, pred_rf)
}

# Model 6: Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)
results['Gradient Boost'] = {
    'R²': r2_score(y_test, pred_gb),
    'MAE': mean_absolute_error(y_test, pred_gb)
}

df_results = pd.DataFrame(results).T
print("\nModel Performance:")
print(df_results.round(3))

best = df_results['R²'].idxmax()
print(f"\nBest: {best} (R²={df_results.loc[best, 'R²']:.3f})")

# ============================================================================
# PART 2: CLUSTERING (Find traffic patterns)
# ============================================================================
print("\n[Part 2] Clustering Analysis...")

# Prepare data for clustering
cluster_features = ['hour', 'duration_min', 'avg_speed_kmh', 'rush']
X_cluster = df[cluster_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("\nFound 4 traffic patterns:")
for i in range(4):
    cluster_data = df[df['cluster'] == i]
    print(f"\nPattern {i+1}:")
    print(f"  Size: {len(cluster_data):,} records")
    print(f"  Avg time: {cluster_data['duration_min'].mean():.1f} min")
    print(f"  Avg speed: {cluster_data['avg_speed_kmh'].mean():.1f} km/h")
    print(f"  Rush hour %: {cluster_data['rush'].mean()*100:.0f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating charts...")

# Chart 1: Model Comparison (better visualization)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

models = df_results.index
x_pos = np.arange(len(models))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

bars1 = ax1.barh(x_pos, df_results['R²'], color=colors, edgecolor='black', linewidth=1.5)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(models)
ax1.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_results['R²']):
    ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)

bars2 = ax2.barh(x_pos, df_results['MAE'], color=colors, edgecolor='black', linewidth=1.5)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(models)
ax2.set_xlabel('Error (minutes)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Error (Lower = Better)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_results['MAE']):
    ax2.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('analysis_models.png', dpi=200, bbox_inches='tight')
print("Saved: analysis_models.png")
plt.close()

# Chart 2: Traffic Patterns (Clustering)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pattern by hour
for i in range(4):
    ax = axes[i//2, i%2]
    cluster_data = df[df['cluster'] == i]
    hourly = cluster_data.groupby('hour')['duration_min'].mean()
    
    ax.plot(hourly.index, hourly.values, marker='o', linewidth=2.5, markersize=8)
    ax.fill_between(hourly.index, hourly.values, alpha=0.3)
    ax.set_title(f'Pattern {i+1} ({len(cluster_data):,} trips)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Time (min)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 4))

plt.suptitle('4 Traffic Patterns Discovered by Clustering', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('analysis_patterns.png', dpi=200, bbox_inches='tight')
print("Saved: analysis_patterns.png")
plt.close()

# Chart 3: Feature Impact Analysis
feature_names = ['Hour', 'Weekday', 'Month', 'Distance', 'Weekend', 
                 'Rush', 'Route', 'Night', 'Morning', 'Afternoon', 'Evening']
importance = rf.feature_importances_

sorted_idx = np.argsort(importance)
pos = np.arange(sorted_idx.shape[0])

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(pos, importance[sorted_idx], color='skyblue', edgecolor='black', linewidth=1.2)

# Color top 3 differently
for i in range(-3, 0):
    bars[i].set_color('#FF6B6B')
    bars[i].set_edgecolor('darkred')

ax.set_yticks(pos)
ax.set_yticklabels(np.array(feature_names)[sorted_idx])
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('What Affects Travel Time Most?', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(importance[sorted_idx]):
    ax.text(v + 0.005, i, f'{v*100:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('analysis_factors.png', dpi=200, bbox_inches='tight')
print("Saved: analysis_factors.png")
plt.close()

# Chart 4: Predictions vs Reality (scatter with density)
sample_size = min(2000, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax1.scatter(y_test.iloc[sample_idx], pred_gb[sample_idx], 
           alpha=0.3, s=20, c='steelblue', edgecolor='none')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2.5, label='Perfect prediction')
ax1.set_xlabel('Actual Time (min)', fontsize=11)
ax1.set_ylabel('Predicted Time (min)', fontsize=11)
ax1.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Error distribution
errors = y_test - pred_gb
ax2.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (0 error)')
ax2.set_xlabel('Prediction Error (min)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis_accuracy.png', dpi=200, bbox_inches='tight')
print("Saved: analysis_accuracy.png")
plt.close()

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print("\n1. Model Performance:")
print(f"   Best model: {best}")
print(f"   Accuracy: {df_results.loc[best, 'R²']*100:.1f}%")
print(f"   Error: ±{df_results.loc[best, 'MAE']:.1f} minutes")

print("\n2. Traffic Patterns Found:")
for i in range(4):
    cluster_data = df[df['cluster'] == i]
    avg_time = cluster_data['duration_min'].mean()
    rush_pct = cluster_data['rush'].mean() * 100
    
    if avg_time > 30:
        pattern_type = "Heavy traffic"
    elif avg_time > 25:
        pattern_type = "Moderate traffic"
    else:
        pattern_type = "Light traffic"
    
    print(f"   Pattern {i+1}: {pattern_type} ({avg_time:.1f} min avg, {rush_pct:.0f}% rush hour)")

print("\n3. Top Factors:")
top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:3]
for i, (feat, imp) in enumerate(top_features, 1):
    print(f"   {i}. {feat}: {imp*100:.1f}%")

print("\n4. Recommendations:")
print("   - Avoid rush hours (8 AM, 6 PM) if possible")
print("   - Use shorter routes when available")
print("   - Travel at night for fastest times")
print("   - Weekends are 8-9% faster than weekdays")

# Save results
df_results.to_csv('ml_results.csv')
cluster_summary = df.groupby('cluster').agg({
    'duration_min': 'mean',
    'avg_speed_kmh': 'mean',
    'rush': lambda x: (x.mean() * 100)
}).round(2)
cluster_summary.to_csv('patterns.csv')

print("\nFiles saved:")
print("- ml_results.csv")
print("- patterns.csv")
print("\nDone!")
