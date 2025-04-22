# Chips Category Analysis for Quantium
# This notebook analyzes transaction and customer data to provide strategic recommendations
# for the Category Manager of a supermarket chain

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Improve visualization appearance
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ----- PART 1: DATA LOADING AND EXPLORATION -----

# For Google Colab, we need to upload the files first
from google.colab import files

# Upload transaction data file
print("Please upload the transaction data file (QVI_transaction_data.csv)...")
uploaded = files.upload()
transaction_data_filename = list(uploaded.keys())[0]

# Upload customer data file
print("Please upload the customer data file (purchase behaviour.csv)...")
uploaded = files.upload()
customer_data_filename = list(uploaded.keys())[0]

# Load transaction data
transaction_data = pd.read_csv(transaction_data_filename)
print("Transaction data shape:", transaction_data.shape)
print("\nTransaction data preview:")
transaction_data.head()

# Load customer data
customer_data = pd.read_csv(customer_data_filename)
print("\nCustomer data shape:", customer_data.shape)
print("\nCustomer data preview:")
customer_data.head()

# ----- PART 2: DATA CLEANING AND PREPARATION -----

# Examining transaction data
print("\nTransaction data info:")
transaction_data.info()

print("\nTransaction data summary statistics:")
transaction_data.describe()

# Check for missing values in transaction data
print("\nMissing values in transaction data:")
print(transaction_data.isnull().sum())

# Check for duplicates in transaction data
print("\nDuplicate rows in transaction data:", transaction_data.duplicated().sum())

# Convert date column to datetime format
transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'])

# Add month and year columns for time-based analysis
transaction_data['MONTH'] = transaction_data['DATE'].dt.month
transaction_data['YEAR'] = transaction_data['DATE'].dt.year
transaction_data['WEEK_OF_YEAR'] = transaction_data['DATE'].dt.isocalendar().week

# Check for outliers in transaction data
plt.figure(figsize=(10, 6))
transaction_data.boxplot(column='TOT_SALES')
plt.title('Boxplot of Total Sales')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()

# Investigating and handling outliers in total sales
q1 = transaction_data['TOT_SALES'].quantile(0.25)
q3 = transaction_data['TOT_SALES'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"\nOutlier thresholds for TOT_SALES: Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")

# Count of transactions considered as outliers
outliers = transaction_data[(transaction_data['TOT_SALES'] < lower_bound) | (transaction_data['TOT_SALES'] > upper_bound)]
print(f"Number of outlier transactions: {len(outliers)}")

# Check if we have negative values for sales (which should be impossible)
negative_sales = transaction_data[transaction_data['TOT_SALES'] <= 0]
print(f"\nNumber of transactions with non-positive sales: {len(negative_sales)}")

# Examining customer data
print("\nCustomer data info:")
customer_data.info()

print("\nCustomer data summary statistics:")
customer_data.describe()

# Check for missing values in customer data
print("\nMissing values in customer data:")
print(customer_data.isnull().sum())

# Check for duplicates in customer data (by LYLTY_CARD_NBR)
print("\nDuplicate customers in customer data:", customer_data.duplicated(subset=['LYLTY_CARD_NBR']).sum())

# ----- PART 3: FEATURE ENGINEERING -----

# Extract brand and pack size from product name
def extract_brand(product_name):
    # List of common chip brands
    brands = ['Smiths', 'Doritos', 'Pringles', 'Kettle', 'Thins', 'Infuzions', 'Red Rock Deli', 'Natural', 'Cheetos', 'Twisties', 'Burger', 'CCs', 'Grain Waves', 'Woolworths', 'Coles']
    
    for brand in brands:
        if brand.lower() in product_name.lower():
            return brand
    
    # Default value if no known brand found
    return 'Other'

def extract_pack_size(product_name):
    # Regular expression to find sizes like '175g' or '150G'
    matches = re.findall(r'(\d+)(?:[gG]|ML)', product_name)
    if matches:
        return int(matches[0])
    else:
        # Default value if no size found
        return np.nan

# Apply these functions to create new columns
transaction_data['BRAND'] = transaction_data['PROD_NAME'].apply(extract_brand)
transaction_data['PACK_SIZE'] = transaction_data['PROD_NAME'].apply(extract_pack_size)

# Display the updated dataframe
print("\nTransaction data after feature extraction:")
transaction_data[['PROD_NAME', 'BRAND', 'PACK_SIZE']].head(10)

# Check the distribution of brands
print("\nBrand distribution:")
brand_counts = transaction_data['BRAND'].value_counts()
print(brand_counts)

# Check the distribution of pack sizes
print("\nPack size distribution:")
pack_size_counts = transaction_data['PACK_SIZE'].value_counts().sort_index()
print(pack_size_counts)

# Visualize the distribution of brands
plt.figure(figsize=(12, 6))
brand_counts.plot(kind='bar')
plt.title('Distribution of Chip Brands')
plt.xlabel('Brand')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize the distribution of pack sizes
plt.figure(figsize=(12, 6))
pack_size_counts.plot(kind='bar')
plt.title('Distribution of Pack Sizes')
plt.xlabel('Pack Size (g)')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ----- PART 4: DATA MERGING -----

# Merge transaction and customer data
merged_data = pd.merge(transaction_data, customer_data, on='LYLTY_CARD_NBR', how='inner')

print("\nMerged data shape:", merged_data.shape)
print("\nMerged data preview:")
merged_data.head()

# Check how many transactions could not be matched with customer data
print(f"\nNumber of transactions in original data: {len(transaction_data)}")
print(f"Number of transactions in merged data: {len(merged_data)}")
print(f"Number of transactions lost in merge: {len(transaction_data) - len(merged_data)}")

# Save the merged data to a CSV file
merged_data.to_csv('merged_chips_data.csv', index=False)
files.download('merged_chips_data.csv')  # This will download the file to your local computer

# ----- PART 5: CUSTOMER SEGMENTATION AND ANALYSIS -----

# Create customer-level metrics
customer_metrics = merged_data.groupby('LYLTY_CARD_NBR').agg(
    total_spend=('TOT_SALES', 'sum'),
    avg_spend_per_transaction=('TOT_SALES', 'mean'),
    transaction_count=('TXN_ID', 'nunique'),
    unique_products=('PROD_NAME', 'nunique'),
    first_purchase=('DATE', 'min'),
    last_purchase=('DATE', 'max')
).reset_index()

# Calculate recency (days since last purchase)
latest_date = merged_data['DATE'].max()
customer_metrics['recency'] = (latest_date - customer_metrics['last_purchase']).dt.days

# Calculate frequency (average days between purchases for customers with multiple purchases)
purchase_dates = merged_data.groupby(['LYLTY_CARD_NBR', 'DATE']).size().reset_index()
purchase_dates = purchase_dates.rename(columns={0: 'count'})

def calculate_avg_days_between(card_nbr):
    dates = purchase_dates[purchase_dates['LYLTY_CARD_NBR'] == card_nbr]['DATE'].sort_values().reset_index(drop=True)
    if len(dates) <= 1:
        return np.nan
    else:
        return np.mean([(dates[i+1] - dates[i]).days for i in range(len(dates)-1)])

customer_metrics['avg_days_between_purchases'] = customer_metrics['LYLTY_CARD_NBR'].apply(calculate_avg_days_between)

# Get brand preferences
brand_preferences = merged_data.groupby(['LYLTY_CARD_NBR', 'BRAND']).size().reset_index()
brand_preferences = brand_preferences.rename(columns={0: 'purchase_count'})
brand_preferences = brand_preferences.sort_values(['LYLTY_CARD_NBR', 'purchase_count'], ascending=[True, False])

# Get the most purchased brand for each customer
top_brands = brand_preferences.groupby('LYLTY_CARD_NBR').first().reset_index()
customer_metrics = pd.merge(customer_metrics, top_brands[['LYLTY_CARD_NBR', 'BRAND']], on='LYLTY_CARD_NBR', how='left')
customer_metrics = customer_metrics.rename(columns={'BRAND': 'favorite_brand'})

# Get pack size preferences
pack_preferences = merged_data.groupby(['LYLTY_CARD_NBR', 'PACK_SIZE']).size().reset_index()
pack_preferences = pack_preferences.rename(columns={0: 'purchase_count'})
pack_preferences = pack_preferences.sort_values(['LYLTY_CARD_NBR', 'purchase_count'], ascending=[True, False])

# Get the most purchased pack size for each customer
top_packs = pack_preferences.groupby('LYLTY_CARD_NBR').first().reset_index()
customer_metrics = pd.merge(customer_metrics, top_packs[['LYLTY_CARD_NBR', 'PACK_SIZE']], on='LYLTY_CARD_NBR', how='left')
customer_metrics = customer_metrics.rename(columns={'PACK_SIZE': 'favorite_pack_size'})

# Add purchase behavior data to customer metrics
customer_metrics = pd.merge(customer_metrics, customer_data, on='LYLTY_CARD_NBR', how='left')

# Print customer metrics
print("\nCustomer metrics:")
customer_metrics.head()

# ----- PART 6: CUSTOMER SEGMENTATION WITH K-MEANS -----

# Preparing data for clustering
# Select numerical features for clustering
features_for_clustering = ['total_spend', 'avg_spend_per_transaction', 'transaction_count', 'recency']
cluster_data = customer_metrics[features_for_clustering].copy()

# Replace NaN values with median
for column in cluster_data.columns:
    cluster_data[column] = cluster_data[column].fillna(cluster_data[column].median())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Choose optimal number of clusters (let's say 4 based on the elbow plot)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_metrics['cluster'] = kmeans.fit_predict(scaled_data)

# Display cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                             columns=features_for_clustering)
print("\nCluster centers:")
print(cluster_centers)

# Count of customers in each cluster
print("\nCustomers per cluster:")
print(customer_metrics['cluster'].value_counts())

# Analyze clusters
cluster_analysis = customer_metrics.groupby('cluster').agg(
    customer_count=('LYLTY_CARD_NBR', 'count'),
    avg_total_spend=('total_spend', 'mean'),
    avg_spend_per_transaction=('avg_spend_per_transaction', 'mean'),
    avg_transaction_count=('transaction_count', 'mean'),
    avg_recency=('recency', 'mean'),
    avg_unique_products=('unique_products', 'mean')
).reset_index()

print("\nCluster analysis:")
print(cluster_analysis)

# Visualize clusters
plt.figure(figsize=(14, 10))
for i, feature in enumerate(features_for_clustering):
    plt.subplot(2, 2, i+1)
    for cluster in range(optimal_k):
        subset = customer_metrics[customer_metrics['cluster'] == cluster]
        plt.hist(subset[feature], alpha=0.5, label=f'Cluster {cluster}')
    plt.title(f'Distribution of {feature} by Cluster')
    plt.legend()
plt.tight_layout()
plt.show()

# ----- PART 7: ANALYZE PURCHASE BEHAVIOR BY SEGMENTS -----

# Merge cluster information back to the transaction data
customer_clusters = customer_metrics[['LYLTY_CARD_NBR', 'cluster']]
transaction_with_clusters = pd.merge(merged_data, customer_clusters, on='LYLTY_CARD_NBR', how='left')

# Analyze spending patterns by cluster
cluster_spending = transaction_with_clusters.groupby('cluster').agg(
    total_sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique'),
    customer_count=('LYLTY_CARD_NBR', 'nunique')
).reset_index()

cluster_spending['sales_per_customer'] = cluster_spending['total_sales'] / cluster_spending['customer_count']
cluster_spending['sales_per_transaction'] = cluster_spending['total_sales'] / cluster_spending['transaction_count']
cluster_spending['transactions_per_customer'] = cluster_spending['transaction_count'] / cluster_spending['customer_count']

print("\nSpending patterns by cluster:")
print(cluster_spending)

# Visualize sales by cluster
plt.figure(figsize=(12, 6))
sns.barplot(x='cluster', y='sales_per_customer', data=cluster_spending)
plt.title('Average Sales per Customer by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Sales per Customer ($)')
plt.tight_layout()
plt.show()

# Analyze brand preferences by cluster
brand_by_cluster = transaction_with_clusters.groupby(['cluster', 'BRAND']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique')
).reset_index()

# Calculate share of sales by brand within each cluster
total_sales_by_cluster = brand_by_cluster.groupby('cluster')['sales'].sum().reset_index()
brand_by_cluster = pd.merge(brand_by_cluster, total_sales_by_cluster, on='cluster', suffixes=('', '_total'))
brand_by_cluster['sales_share'] = brand_by_cluster['sales'] / brand_by_cluster['sales_total']

# Get top brands by sales for each cluster
top_brands_by_cluster = brand_by_cluster.sort_values(['cluster', 'sales'], ascending=[True, False])
print("\nTop brands by sales in each cluster:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(top_brands_by_cluster[top_brands_by_cluster['cluster'] == cluster].head(5))

# Visualize top brands for each cluster
plt.figure(figsize=(16, 12))
for i in range(optimal_k):
    plt.subplot(2, 2, i+1)
    cluster_data = top_brands_by_cluster[top_brands_by_cluster['cluster'] == i].head(5)
    sns.barplot(x='sales', y='BRAND', data=cluster_data)
    plt.title(f'Top 5 Brands for Cluster {i}')
plt.tight_layout()
plt.show()

# Analyze pack size preferences by cluster
packsize_by_cluster = transaction_with_clusters.groupby(['cluster', 'PACK_SIZE']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique')
).reset_index()

# Calculate share of sales by pack size within each cluster
packsize_by_cluster = pd.merge(packsize_by_cluster, total_sales_by_cluster, on='cluster', suffixes=('', '_total'))
packsize_by_cluster['sales_share'] = packsize_by_cluster['sales'] / packsize_by_cluster['sales_total']

# Get top pack sizes by sales for each cluster
top_packsizes_by_cluster = packsize_by_cluster.sort_values(['cluster', 'sales'], ascending=[True, False])
print("\nTop pack sizes by sales in each cluster:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(top_packsizes_by_cluster[top_packsizes_by_cluster['cluster'] == cluster].head(5))

# Visualize pack size preferences for each cluster
plt.figure(figsize=(16, 12))
for i in range(optimal_k):
    plt.subplot(2, 2, i+1)
    cluster_data = top_packsizes_by_cluster[top_packsizes_by_cluster['cluster'] == i].head(5)
    sns.barplot(x='sales', y='PACK_SIZE', data=cluster_data)
    plt.title(f'Top 5 Pack Sizes for Cluster {i}')
plt.tight_layout()
plt.show()

# Analyze life stage and premium customer status by cluster
lifestage_by_cluster = transaction_with_clusters.groupby(['cluster', 'LIFESTAGE']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique'),
    customer_count=('LYLTY_CARD_NBR', 'nunique')
).reset_index()

lifestage_by_cluster = pd.merge(lifestage_by_cluster, total_sales_by_cluster, on='cluster', suffixes=('', '_total'))
lifestage_by_cluster['sales_share'] = lifestage_by_cluster['sales'] / lifestage_by_cluster['sales_total']

premium_by_cluster = transaction_with_clusters.groupby(['cluster', 'PREMIUM_CUSTOMER']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique'),
    customer_count=('LYLTY_CARD_NBR', 'nunique')
).reset_index()

premium_by_cluster = pd.merge(premium_by_cluster, total_sales_by_cluster, on='cluster', suffixes=('', '_total'))
premium_by_cluster['sales_share'] = premium_by_cluster['sales'] / premium_by_cluster['sales_total']

print("\nLifestage distribution by cluster:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(lifestage_by_cluster[lifestage_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False))

print("\nPremium customer distribution by cluster:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(premium_by_cluster[premium_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False))

# Visualize lifestage by cluster
plt.figure(figsize=(14, 10))
for i in range(optimal_k):
    plt.subplot(2, 2, i+1)
    cluster_data = lifestage_by_cluster[lifestage_by_cluster['cluster'] == i]
    sns.barplot(x='sales', y='LIFESTAGE', data=cluster_data)
    plt.title(f'Lifestage Sales for Cluster {i}')
plt.tight_layout()
plt.show()

# ----- PART 8: TIME-BASED ANALYSIS -----

# Analyze sales trends over time
monthly_sales = transaction_data.groupby(['YEAR', 'MONTH']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique')
).reset_index()

monthly_sales['year_month'] = monthly_sales['YEAR'].astype(str) + '-' + monthly_sales['MONTH'].astype(str).str.zfill(2)
monthly_sales = monthly_sales.sort_values(['YEAR', 'MONTH'])

plt.figure(figsize=(12, 6))
sns.lineplot(x='year_month', y='sales', data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze weekly sales trends
weekly_sales = transaction_data.groupby(['YEAR', 'WEEK_OF_YEAR']).agg(
    sales=('TOT_SALES', 'sum'),
    transaction_count=('TXN_ID', 'nunique')
).reset_index()

weekly_sales['year_week'] = weekly_sales['YEAR'].astype(str) + '-W' + weekly_sales['WEEK_OF_YEAR'].astype(str).str.zfill(2)
weekly_sales = weekly_sales.sort_values(['YEAR', 'WEEK_OF_YEAR'])

plt.figure(figsize=(16, 6))
sns.lineplot(x='year_week', y='sales', data=weekly_sales)
plt.title('Weekly Sales Trend')
plt.xlabel('Year-Week')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ----- PART 9: COMPILE KEY INSIGHTS AND RECOMMENDATIONS -----

# Define a summary function to describe each cluster based on its characteristics
def describe_cluster(cluster_num, cluster_analysis, top_brands, top_packsizes, lifestage_data, premium_data):
    cluster_row = cluster_analysis[cluster_analysis['cluster'] == cluster_num].iloc[0]
    
    top_brand = top_brands[top_brands['cluster'] == cluster_num].iloc[0]['BRAND']
    top_pack = top_packsizes[top_packsizes['cluster'] == cluster_num].iloc[0]['PACK_SIZE']
    
    top_lifestage = lifestage_data[lifestage_data['cluster'] == cluster_num].sort_values('sales', ascending=False).iloc[0]['LIFESTAGE']
    top_premium = premium_data[premium_data['cluster'] == cluster_num].sort_values('sales', ascending=False).iloc[0]['PREMIUM_CUSTOMER']
    
    description = f"Cluster {cluster_num}:\n"
    description += f"- {cluster_row['customer_count']} customers ({cluster_row['customer_count'] / cluster_analysis['customer_count'].sum():.1%} of total)\n"
    description += f"- Average spend: ${cluster_row['avg_total_spend']:.2f} per customer\n"
    description += f"- Average transactions: {cluster_row['avg_transaction_count']:.1f} per customer\n"
    description += f"- Favorite brand: {top_brand}\n"
    description += f"- Preferred pack size: {top_pack}g\n"
    description += f"- Dominant lifestage: {top_lifestage}\n"
    description += f"- Premium customer status: {top_premium}\n"
    
    return description

print("\n----- CUSTOMER SEGMENT PROFILES -----")
for cluster in range(optimal_k):
    print(describe_cluster(
        cluster,
        cluster_analysis,
        top_brands_by_cluster.groupby('cluster').first().reset_index(),
        top_packsizes_by_cluster.groupby('cluster').first().reset_index(),
        lifestage_by_cluster,
        premium_by_cluster
    ))
    print()

print("\n----- KEY INSIGHTS -----")
print("1. Customer Segmentation:")
print(cluster_analysis)

print("\n2. Spending Patterns by Segment:")
print(cluster_spending)

print("\n3. Top Brands by Segment:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(top_brands_by_cluster[top_brands_by_cluster['cluster'] == cluster].head(3)[['BRAND', 'sales', 'sales_share']])

print("\n4. Pack Size Preferences by Segment:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster}:")
    print(top_packsizes_by_cluster[top_packsizes_by_cluster['cluster'] == cluster].head(3)[['PACK_SIZE', 'sales', 'sales_share']])

print("\n5. Customer Demographics by Segment:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} - Top Lifestages:")
    print(lifestage_by_cluster[lifestage_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False).head(3)[['LIFESTAGE', 'sales', 'sales_share']])
    print(f"\nCluster {cluster} - Premium Status:")
    print(premium_by_cluster[premium_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False)[['PREMIUM_CUSTOMER', 'sales', 'sales_share']])

# Save customer segments for future use
customer_metrics.to_csv('customer_segments.csv', index=False)
files.download('customer_segments.csv')  # This will download the file to your local computer

print("\n----- STRATEGIC RECOMMENDATIONS -----")
print("Based on the analysis, here are the strategic recommendations for Julia:")

# Generate targeted recommendations based on actual analysis results
high_value_cluster = cluster_spending.sort_values('sales_per_customer', ascending=False).iloc[0]['cluster']
high_frequency_cluster = cluster_spending.sort_values('transactions_per_customer', ascending=False).iloc[0]['cluster']

print(f"\n1. Focus on high-value customer segment (Cluster {high_value_cluster}):")
print(f"   - These customers spend ${cluster_spending[cluster_spending['cluster'] == high_value_cluster].iloc[0]['sales_per_customer']:.2f} on average")
print(f"   - Develop loyalty programs to retain these valuable customers")
print(f"   - Consider premium product offerings that appeal to this segment")

print(f"\n2. Increase purchase frequency for Cluster {high_frequency_cluster}:")
print(f"   - These customers already buy frequently ({cluster_spending[cluster_spending['cluster'] == high_frequency_cluster].iloc[0]['transactions_per_customer']:.1f} transactions per customer)")
print(f"   - Create bundle offers to increase basket size")
print(f"   - Consider targeted promotions based on their favorite brands and pack sizes")

print("\n3. Product mix optimization:")
print("   - Ensure top brands for each segment are prominently displayed in stores")
print("   - Adjust pack size offerings based on segment preferences")
print("   - Consider introducing new pack sizes based on segment-specific needs")

print("\n4. Targeted marketing campaigns:")
for cluster in range(optimal_k):
    top_lifestage = lifestage_by_cluster[lifestage_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False).iloc[0]['LIFESTAGE']
    top_premium = premium_by_cluster[premium_by_cluster['cluster'] == cluster].sort_values('sales', ascending=False).iloc[0]['PREMIUM_CUSTOMER']
    print(f"   - Cluster {cluster}: Target {top_lifestage} customers who are {top_premium}")

print("\n5. Price sensitivity analysis:")
print("   - Conduct further analysis to understand price elasticity by segment")
print("   - Develop pricing strategies that optimize revenue while maintaining segment-specific value propositions")

# Save figures for the report
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='sales_per_customer', data=cluster_spending)
plt.title('Average Sales per Customer by Cluster')
plt.xlabel('Customer Segment')
plt.ylabel('Average Sales per Customer ($)')
plt.tight_layout()
plt.savefig('sales_by_segment.png')
files.download('sales_by_segment.png')

plt.figure(figsize=(12, 8))
for i, feature in enumerate(['total_spend', 'transaction_count']):
    plt.subplot(1, 2, i+1)
    for cluster in range(optimal_k):
        subset = customer_metrics[customer_metrics['cluster'] == cluster]
        plt.hist(subset[feature], alpha=0.5, label=f'Segment {cluster}')
    plt.title(f'Distribution of {feature} by Segment')
    plt.legend()
plt.tight_layout()
plt.savefig('segment_distributions.png')
files.download('segment_distributions.png')

# Create a summary report
print("\n----- EXECUTIVE SUMMARY FOR JULIA -----")
print("Our analysis of the chips category has identified distinct customer segments with unique purchasing behaviors.")
print("We recommend a targeted approach for the next half year that focuses on:")
print("1. Retention strategies for high-value customers")
print("2. Increasing purchase frequency across segments")
print("3. Optimizing product mix based on segment preferences")
print("4. Implementing targeted marketing campaigns tailored to each segment")
print("5. Developing segment-specific pricing strategies")
print("\nThese recommendations are designed to drive growth in the chips category by addressing the specific needs and behaviors of each customer segment.")
