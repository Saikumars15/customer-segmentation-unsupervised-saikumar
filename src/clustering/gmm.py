#Read the dataset
import pandas as pd
df = pd.read_csv('customer-segmentation-unsupervised-saikumar\\src\\used_cars_data.csv')

#Display basic information about the dataset
df.info()

#Check for describing values
df.describe()

#check for missing values
df.isnull().sum()

# Handle missing values by filling with mode (most frequent value) for categorical columns
df.fillna(df.mode().iloc[0], inplace=True)

#Recheck for missing values after imputation
df.isnull().sum()

#Check for duplicates
df.duplicated().sum()

#Remove duplicates if any
df.drop_duplicates(inplace=True)

#Check for outliers using IQR method
import numpy as np
from IPython.display import display
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Column: {col}, outliers: {len(outliers)}, lower={lower:.3f}, upper={upper:.3f}")
    if len(outliers):
        display(outliers[['S.No.', 'Name', col]].sort_values(col).head(10))


# After identifying outliers, we can choose to remove them or cap them. Here, we will remove them.
# convert extractable numeric fields first
df['Mileage_num'] = df['Mileage'].str.extract(r'([\d.]+)').astype(float)
df['Engine_cc'] = df['Engine'].str.extract(r'(\d+)').astype(float)
df['Power_bhp'] = df['Power'].str.extract(r'([\d.]+)').astype(float)
df['New_Price_num'] = df['New_Price'].str.replace('[^0-9.]', '', regex=True).astype(float)

# use all numeric columns (including newly created ones)
all_numeric_cols = df.select_dtypes(include=[np.number]).columns

mask = pd.Series(True, index=df.index)
for c in all_numeric_cols:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask &= df[c].between(lower, upper, inclusive='both')

df = df[mask].copy()

print("After outlier removal:", df.shape)
df[all_numeric_cols].describe()


#Checking for Object columns and their unique values
df.info()

#Creating dummies for Categorical columns and their unique values
cat_cols = df.select_dtypes(include=['object', 'category']).columns
print("Categorical columns to dummy:", list(cat_cols))

df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=False)

print("Dummy dataframe shape:", df_dummies.shape)
print("Dummy columns sample:", df_dummies.columns[:50].tolist())

df_dummies.head()

#dropping dummies.
df = pd.get_dummies(df, drop_first=True)

#Cross-checking the final dataframe info after preprocessing
df.info()

#Feature engineering and Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

import datetime

# Use df_dummies (contains original Year, Price and one-hot Location)
location_cols = [c for c in df_dummies.columns if c.startswith('Location_')]
df_rfm = df_dummies.copy()

# recover textual location from one-hot
df_rfm['Location'] = df_rfm[location_cols].idxmax(axis=1).str.replace('Location_', '', regex=False)

# reference date for recency (choose a fixed analysis date)
reference_date = datetime.datetime(2025, 1, 1)

# compute recency as age of the car in years (lower = more recent)
df_rfm['Recency'] = reference_date.year - df_rfm['Year']

# RFM aggregation by Location
rfm = df_rfm.groupby('Location').agg(
    Recency=('Recency', 'min'),          # most recent car in that location
    Frequency=('S.No.', 'count'),        # number of listings
    Monetary=('Price', 'mean')           # average price
).reset_index()

# score each dimension into 1..5
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=False, duplicates='drop')
rfm['R_score'] = 5 - rfm['R_score']   # smaller recency is better
rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=False, duplicates='drop') + 1
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=False, duplicates='drop') + 1

rfm['RFM_Score'] = rfm['R_score'] * 100 + rfm['F_score'] * 10 + rfm['M_score']

# simple segment labeling
def rfm_segment(row):
    if row['RFM_Score'] >= 445:
        return 'Best'
    if row['RFM_Score'] >= 350:
        return 'Loyal'
    if row['RFM_Score'] >= 250:
        return 'Potential'
    return 'Needs Attention'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)

rfm.sort_values('RFM_Score', ascending=False).head(20)




rfm['AvgValue'] = rfm['Monetary'] / rfm['Frequency']


rfm['Value_per_day'] = rfm['Monetary'] / (rfm['Recency'] + 1)


print(df.dtypes)


df.info()



#Exploratory Data Analysis (EDA)
#Univariate Analysis
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(rfm['Monetary'])
plt.show()

#Bivariate Analysis
sns.scatterplot(x='Frequency', y='Monetary', data=rfm)
plt.show()


#Multivariate Analysis for numerical columns
import seaborn as sns

import matplotlib.pyplot as plt

# Select only numeric columns for correlation
numeric_rfm = rfm.select_dtypes(include=[np.number])
sns.heatmap(numeric_rfm.corr(), annot=True)
plt.show()


#Multivariate Analysis for all numeric and boolean columns in df
import seaborn as sns

import matplotlib.pyplot as plt

# Compute correlation matrix for all numeric and boolean columns
corr_matrix = df.select_dtypes(include=[np.number, 'bool']).corr()

# Plot heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of All Numeric and Boolean Columns in df')
plt.show()


#Clustering Analysis using K-Means
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Apply K-Means clustering on the scaled data
kmeans = KMeans(n_clusters=4, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(scaled_data)

# Display cluster counts
print(df['KMeans_Cluster'].value_counts())

# Visualize clusters using PCA for dimensionality reduction
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(8, 6))
for cluster in range(4):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
plt.title('K-Means Clusters (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()



#Clustering Analysis using Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(scaled_data, method='ward')
dendrogram(linked)
plt.show()


#Clustering Analysis using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_data)

# Display cluster counts (including noise as -1)
print(df['DBSCAN_Cluster'].value_counts())

# Visualize DBSCAN clusters using existing PCA components
plt.figure(figsize=(8, 6))
for cluster in sorted(set(df['DBSCAN_Cluster'])):
    cluster_data = df[df['DBSCAN_Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}' if cluster != -1 else 'Noise')
plt.title('DBSCAN Clusters (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()


#Clustering Analysis using Gaussian Mixture Models (GMM)
from sklearn.mixture import GaussianMixture

# Apply Gaussian Mixture Model clustering to the scaled data
gmm = GaussianMixture(n_components=4, random_state=42)
df['GMM_Cluster'] = gmm.fit_predict(scaled_data)

# Display cluster counts
print(df['GMM_Cluster'].value_counts())

# Visualize GMM clusters using existing PCA components
plt.figure(figsize=(8, 6))
for cluster in range(4):
    cluster_data = df[df['GMM_Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
plt.title('GMM Clusters (PCA Reduced)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()