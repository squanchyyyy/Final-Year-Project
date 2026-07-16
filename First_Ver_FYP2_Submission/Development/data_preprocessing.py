import os
import numpy as np
import pandas as pd
from scipy.io import arff as scipy_arff
from sklearn.preprocessing import RobustScaler
import arff as liac_arff  # liac-arff for writing

# Step 1: Load dataset
file_path = '../FYP2/Development/jm1.arff'
data, meta = scipy_arff.loadarff(file_path)
df = pd.DataFrame(data)

# Step 2: Decode byte strings in the 'defects' column
df['defects'] = df['defects'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Step 3: Map string labels to numeric
df['defects'] = df['defects'].map({'false': 0, 'true': 1})

# Step 4: Handle missing values
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"Missing values detected: {missing}. Filling with median values...")
    df = df.fillna(df.median(numeric_only=True))
else:
    print("No missing values detected.")

# Step 5: Split features and label
X = df.drop(columns=['defects'])
y = df['defects']

# Step 6: Feature scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Reconstruct DataFrame
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['defects'] = y.values

# Step 8: Define ARFF metadata
attributes = [(col, 'NUMERIC') for col in X.columns]
attributes.append(('defects', ['0', '1']))

arff_dict = {
    'description': 'Preprocessed JM1 dataset with median imputation and robust scaling',
    'relation': 'jm1_preprocessed',
    'attributes': attributes,
    'data': processed_df.values.tolist()
}

# Step 9: Save to new ARFF file
output_path = os.path.join(os.getcwd(), 'jm1_preprocessed.arff')
with open(output_path, 'w') as f:
    liac_arff.dump(arff_dict, f)

print(f"✅ Preprocessed ARFF file saved to: {output_path}")