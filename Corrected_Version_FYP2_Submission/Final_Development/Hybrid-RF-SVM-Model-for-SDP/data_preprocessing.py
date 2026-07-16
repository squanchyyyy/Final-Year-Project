import os
import numpy as np
import pandas as pd
from scipy.io import arff 
from sklearn.preprocessing import RobustScaler
import arff as liac_arff  # liac-arff for writing

# Step 1: Load dataset
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

# Step 2: Decode byte strings in all object columns
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Step 3: Save original defects column
label_col = df['defects']  # still contains 'false'/'true'

# Step 4: Handle missing values
if df.drop(columns=['defects']).isnull().sum().sum() > 0:
    print("Missing values detected. Filling with median...")
    df = df.fillna(df.median(numeric_only=True))
else:
    print("No missing values detected.")

# Step 5: Scale only numeric features
X = df.drop(columns=['defects'])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Rebuild DataFrame
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['defects'] = label_col.values  # retain original 'false'/'true'

# Step 7: Define ARFF metadata
attributes = [(col, 'NUMERIC') for col in X.columns]
attributes.append(('defects', ['false', 'true']))

arff_dict = {
    'description': 'Preprocessed JM1 dataset with robust scaling and original labels',
    'relation': 'jm1_preprocessed',
    'attributes': attributes,
    'data': processed_df.values.tolist()
}

# Step 8: Save to ARFF
output_path = os.path.join(os.getcwd(), 'jm1_preprocessed.arff')
with open(output_path, 'w') as f:
    liac_arff.dump(arff_dict, f)

print(f"✅ Preprocessed ARFF file saved to: {output_path}")
