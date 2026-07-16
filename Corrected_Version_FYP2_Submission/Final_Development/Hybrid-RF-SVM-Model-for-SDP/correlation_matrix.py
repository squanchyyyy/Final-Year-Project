import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# Load the ARFF dataset
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Rename the class column to 'defects' if it's named differently
df.rename(columns={'class_attribute': 'defects'}, inplace=True)

# Decode all object (byte-string) columns to regular strings
for column in df.select_dtypes([object]).columns:
    df[column] = df[column].apply(lambda x: x.decode(
        'utf-8') if isinstance(x, bytes) else x)

# Convert 'defects' from 'false'/'true' to 0/1
df['defects'] = df['defects'].map({'false': 0, 'true': 1})

# Convert all applicable columns to numeric (just in case)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix for NASA JM1 Dataset')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#try later
"""import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_prepare_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Standardize the target column name
    if 'class_attribute' in df.columns:
        df.rename(columns={'class_attribute': 'defects'}, inplace=True)
    elif 'class' in df.columns:
        df.rename(columns={'class': 'defects'}, inplace=True)

    # Convert target to 0/1
    if 'defects' in df.columns:
        df['defects'] = df['defects'].astype(str).str.lower().map({'false': 0, 'true': 1})

    # Convert all columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop missing values
    df.dropna(inplace=True)

    return df

def plot_correlation_matrix(df, dataset_name='Dataset'):
    # Compute correlation matrix (absolute correlation helps spot strong relationships)
    corr = df.corr().abs()

    # Mask the upper triangle for a cleaner look
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(16, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5,
                cbar_kws={"shrink": .8}, square=True)
    plt.title(f'🔗 Correlation Matrix - {dataset_name}', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Example usage
file_path = '../FYP2/1. Data Collection/cm1.arff'
df = load_and_prepare_arff(file_path)
plot_correlation_matrix(df, dataset_name='NASA CM1')"""
