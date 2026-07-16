import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import time

start_time = time.time()

# 1️⃣ Load & Preprocess Data
data, meta = arff.loadarff('jm1_preprocessed.arff')
df = pd.DataFrame(data)
df.rename(columns={'defects': 'label'}, inplace=True)

# Decode byte strings and map labels
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df['label'] = df['label'].map({'false': 0, 'true': 1})
df.dropna(inplace=True)

print("Data shape after processing:", df.shape)

# 2️⃣ Split Features & Labels
X = df.drop('label', axis=1)
y = df['label']

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Scale Features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6️⃣ Evaluate Model on Test Set
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Save predictions
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)

# Display Class-wise Metrics
print("\nClass 0 Metrics (Non-Defective Cases):")
print(f"  Precision: {round(report['0']['precision'], 4)}")
print(f"  Recall:    {round(report['0']['recall'], 4)}")
print(f"  F1-Score:  {round(report['0']['f1-score'], 4)}")

print("\nClass 1 Metrics (Defective Cases):")
print(f"  Precision: {round(report['1']['precision'], 4)}")
print(f"  Recall:    {round(report['1']['recall'], 4)}")
print(f"  F1-Score:  {round(report['1']['f1-score'], 4)}")

print("\nOverall ROC-AUC Score:", roc_auc)
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, digits=4))


# 7️⃣ Predict Total Defects in Entire Dataset
X_scaled_full = scaler.transform(X)
full_pred_proba = rf_model.predict_proba(X_scaled_full)[:, 1]
full_pred_labels = (full_pred_proba >= 0.5).astype(int)
total_defects_predicted = np.sum(full_pred_labels)

# 🔢 Breakdown of predicted vs. actual counts
predicted_non_defective = np.sum(full_pred_labels == 0)
predicted_defective = np.sum(full_pred_labels == 1)
actual_non_defective = np.sum(df['label'] == 0)
actual_defective = np.sum(df['label'] == 1)

print("\n📊 Defect Classification Summary (Full Dataset):")
print(f"1️⃣ Predicted non-defective modules: {predicted_non_defective} out of {len(df)}")
print(f"2️⃣ Actual non-defective modules:    {actual_non_defective} out of {len(df)}")
print(f"3️⃣ Predicted defective modules:     {predicted_defective} out of {len(df)}")
print(f"4️⃣ Actual defective modules:        {actual_defective} out of {len(df)}")
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

"""START

1. LOAD DATA
   - Read .arff file into DataFrame
   - Rename 'defects' → 'label'
   - Convert byte strings and map 'false'→0, 'true'→1
   - Drop rows with missing values

2. SPLIT into features (X) and labels (y)

3. SPLIT into training and test sets (80:20)

4. SCALE FEATURES
   - Use RobustScaler to reduce impact of outliers

5. TRAIN Random Forest on training data

6. EVALUATE on test set
   - Predict probabilities → class labels (threshold = 0.5)
   - Print precision, recall, F1-score, ROC-AUC

7. PREDICT FULL DATASET
   - Apply scaler to full dataset
   - Predict defect probabilities
   - Convert to final predictions (threshold = 0.5)
   - Count total predicted defects in entire dataset

END"""