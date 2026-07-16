import pandas as pd
from scipy.io import arff

# Load the dataset
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Decode byte strings (common in ARFF datasets)
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# Rename 'class_attribute' to 'class' if necessary
df.rename(columns={'class_attribute': 'class'}, inplace=True)

# Standardize target column (if exists)
if 'class' in df.columns:
    df['defects'] = df['class'].astype(str).str.lower().map({'false': 0, 'true': 1})

# Show all columns
pd.set_option('display.max_columns', None)

# Summary statistics for numeric features
print("📊 Numeric Feature Summary:\n")
print(df.describe())

# Summary statistics for categorical features (like 'class')
print("\n🔤 Categorical Feature Summary:\n")
print(df.describe(include=['object', 'bool']))
