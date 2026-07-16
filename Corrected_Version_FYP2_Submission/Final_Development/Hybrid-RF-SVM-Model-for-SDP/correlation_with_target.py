import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt

# Load ARFF
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Decode byte strings
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].apply(lambda x: x.decode(
        'utf-8') if isinstance(x, bytes) else x)

# Convert 'defects' to binary
df['defects'] = df['defects'].map({'false': 0, 'true': 1}).astype(int)

# Convert all to numeric
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Compute correlation with the target
correlation = df.corr(numeric_only=True)['defects'].drop(
    'defects')  # Drop self-correlation

# Sort by absolute value of correlation
correlation_sorted = correlation.abs().sort_values(ascending=False)

# Display top features
print("Correlation of features with 'defects':")
print(correlation_sorted)

# Optional: Plot heatmap of top N features
top_n = 10
top_features = correlation_sorted.head(top_n).index.tolist() + ['defects']
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Top Features with Target)')
plt.show()
