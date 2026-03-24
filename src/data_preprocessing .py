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






