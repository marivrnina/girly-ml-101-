import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
os.chdir('/Users/mshvrnina/Desktop')

df = pd.read_csv('Fortune_1000.csv')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '', regex=False)
print("\nCleaned columns:")
print(list(df.columns))

# Convert text to boolean
df['profitable'] = df['profitable'].map({'yes': True, 'no': False})
df['ceo_woman'] = df['ceo_woman'].map({'yes': True, 'no': False})
df['ceo_founder']= df['ceo_founder'].map({'yes': True, 'no': False})

df['profit_margin']= (df['profit'] / df['revenue']) * 100

df = df.drop_duplicates()
print(f"\nShape after cleaning: {df.shape}")

# Top 10 by revenue
top10_rev= df.nlargest(10, 'revenue')[['company', 'revenue']].sort_values('revenue')
print("\nTop 10 by Revenue:")
print(top10_rev)

plt.figure(figsize=(10, 6))
plt.barh(top10_rev['company'], top10_rev['revenue'])
plt.title('Top 10 Companies by Revenue')
plt.xlabel('Revenue (millions)')
plt.tight_layout()
plt.savefig('top10_revenue.png', dpi=150, bbox_inches='tight')
plt.show()

# Top 10 by profit
top10_pft = df.nlargest(10, 'profit')[['company', 'profit']].sort_values('profit')
print("\nTop 10 by Profit:")
print(top10_pft)

plt.figure(figsize=(10, 6))
plt.barh(top10_pft['company'], top10_pft['profit'])
plt.title('Top 10 Companies by Profit')
plt.xlabel('Profit (millions)')
plt.tight_layout()
plt.savefig('top10_profit.png', dpi=150, bbox_inches='tight')
plt.show()


# Number of companies per sector
sector_counts = df['sector'].value_counts().head(10)
print("\nCompanies per Sector:")
print(sector_counts)

plt.figure(figsize=(12, 6))
plt.bar(sector_counts.index, sector_counts.values)
plt.title('Number of Companies by Sector')
plt.xlabel('Sector')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('sector_counts.png', dpi=150, bbox_inches='tight')
plt.show()

sector_prof = df.groupby('sector')['profitable'].mean().sort_values(ascending=False) * 100
print(f"\nProfitability by Sector:")
print(sector_prof.head(10))

plt.figure(figsize=(12, 6))
plt.bar(sector_prof.index, sector_prof.values)
plt.title('Profitability Rate by Sector (%)')
plt.ylabel('% Profitable')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('sector_profitability.png', dpi=150, bbox_inches='tight')
plt.show()


# Top states
top_states = df['state'].value_counts().head(12)
print("\nTop States by HQ Count:")
print(top_states)

plt.figure(figsize=(10, 6))
plt.bar(top_states.index, top_states.values)
plt.title('Top 12 States by Number of HQs')
plt.xlabel('State')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('top_states.png', dpi=150, bbox_inches='tight')
plt.show()

# Top cities
top_cities = df['city'].value_counts().head(10)
print("\nTop Cities by HQ Count:")
print(top_cities)

plt.figure(figsize=(10, 6))
plt.barh(top_cities.sort_values().index, top_cities.sort_values().values)
plt.title('Top 10 Cities by Number of HQs')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('top_cities.png', dpi=150, bbox_inches='tight')
plt.show()


women_pct = df['ceo_woman'].mean() * 100
founder_pct = df['ceo_founder'].mean() * 100

print(f"Women CEOs: {women_pct:.1f}%")
print(f"Founder CEOs: {founder_pct:.1f}%")

# Women CEO pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    [women_pct, 100 - women_pct],
    labels=['Women CEO', 'Men CEO'],
    autopct='%1.1f%%',
    startangle=90
)
plt.title('CEO Gender Distribution')
plt.tight_layout()
plt.savefig('ceo_gender.png', dpi=150, bbox_inches='tight')
plt.show()

# Performance comparison
women_margin = df[df['ceo_woman'] == True]['profit_margin'].median()
men_margin = df[df['ceo_woman'] == False]['profit_margin'].median()
founder_margin = df[df['ceo_founder'] == True]['profit_margin'].median()
non_founder_margin = df[df['ceo_founder'] == False]['profit_margin'].median()

print(f"\nMedian profit margin - Women CEOs: {women_margin:.2f}%")
print(f"Median profit margin - Men CEOs: {men_margin:.2f}%")
print(f"Median profit margin - Founder CEOs: {founder_margin:.2f}%")
print(f"Median profit margin - Non-founder: {non_founder_margin:.2f}%")

categories = ['Women CEO', 'Men CEO', 'Founder CEO', 'Non-Founder']
values = [women_margin, men_margin, founder_margin, non_founder_margin]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['#ff6b9d', '#4ecdc4', '#45b7d1', '#96ceb4'])
plt.title('Median Profit Margin by CEO Type')
plt.ylabel('Profit Margin (%)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('ceo_profit_margin.png', dpi=150, bbox_inches='tight')
plt.show()


# Correlation matrix
numeric_cols = ['rank', 'revenue', 'profit', 'employees', 'market_cap', 'profit_margin']
corr = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Revenue vs Profit scatter
plt.figure(figsize=(10, 6))
colors = ['#ff6b9d' if x else '#4ecdc4' for x in df['profitable']]
plt.scatter(df['revenue'], df['profit'], alpha=0.6, c=colors)
plt.title('Revenue vs Profit')
plt.xlabel('Revenue (millions)')
plt.ylabel('Profit (millions)')
plt.tight_layout()
plt.savefig('revenue_vs_profit.png', dpi=150, bbox_inches='tight')
plt.show()

print("KEY FINDINGS SUMMARY")
print(f"Total companies analyzed: {len(df)}")
print(f"Total revenue (all): ${df['revenue'].sum()/1e6:.2f}T")
print(f"Overall profitability rate: {df['profitable'].mean()*100:.1f}%")
print(f"Women CEOs: {women_pct:.1f}%")
print(f"Founder-led companies: {founder_pct:.1f}%")
print(f"Top sector by count: {sector_counts.index[0]}")
print(f"Top state by HQ count: {top_states.index[0]}")
print(f"Top city by HQ count: {top_cities.index[0]}")

