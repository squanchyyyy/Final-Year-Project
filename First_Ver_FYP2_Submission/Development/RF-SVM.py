import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import time 

start_time = time.time()

# 1. Load & Preprocess Data
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)
df.rename(columns={'defects': 'label'}, inplace=True)

# Decode binary columns
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(lambda x: x.decode(
        'utf-8') if isinstance(x, bytes) else x)
df['label'] = df['label'].map({'false': 0, 'true': 1})
df.dropna(inplace=True)
print("Data shape after processing:", df.shape)

# 2. Split Features & Labels
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Scale Features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Base Models
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

rf_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)

# 5. Get Probabilities from Base Models
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# 6. Stack Predictions as Features for Meta-Model
stacked_features = np.column_stack((rf_proba, svm_proba))

# 7. Train Meta-Model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)
final_proba = meta_model.predict_proba(stacked_features)[:, 1]
final_pred = (final_proba >= 0.5).astype(int)


# 8. Evaluation
report = classification_report(y_test, final_pred, output_dict=True)
roc_auc = round(roc_auc_score(y_test, final_proba), 4)

print("\nClass 0 Metrics:")
print(f"  Precision: {round(report['0']['precision'], 4)}")
print(f"  Recall:    {round(report['0']['recall'], 4)}")
print(f"  F1-Score:  {round(report['0']['f1-score'], 4)}")

print("\nClass 1 Metrics (Defective Cases):")
print(f"  Precision: {round(report['1']['precision'], 4)}")
print(f"  Recall:    {round(report['1']['recall'], 4)}")
print(f"  F1-Score:  {round(report['1']['f1-score'], 4)}")

print("\nOverall ROC-AUC Score:", roc_auc)
print("\nFull Classification Report:")
print(classification_report(y_test, final_pred, digits=4))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
"""Pseudocode
START

1. LOAD and PREPROCESS the dataset
   - Load ARFF data and convert to DataFrame
   - Decode binary labels from string to integer (0 or 1)
   - Drop missing values

2. SPLIT the dataset
   - Features (X) = all columns except 'label'
   - Labels (y) = 'label' column
   - Split into training and test sets (e.g., 80/20)

3. SCALE feature values using RobustScaler
   - Fit on training data and transform both training and test sets

4. TRAIN base models
   - Train Random Forest on scaled training data
   - Train SVM (RBF kernel, probability=True) on scaled training data

5. PREDICT class probabilities using both models on test data
   - Get probability outputs from both RF and SVM

6. STACK predictions
   - Combine RF and SVM probability outputs as new input features

7. TRAIN meta-model (Logistic Regression)
   - Use the stacked predictions as input and actual labels for training

8. PREDICT final outputs using the meta-model
   - Classify test instances based on meta-model probabilities

9. EVALUATE the final predictions
   - Calculate precision, recall, F1-score for each class
   - Compute overall ROC-AUC score
   - Display classification report

END
"""
