import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Decode and map class labels
df['label'] = df['defects'].apply(lambda x: 1 if x == b'true' else 0)
df.drop('defects', axis=1, inplace=True)

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Extract feature importances
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
