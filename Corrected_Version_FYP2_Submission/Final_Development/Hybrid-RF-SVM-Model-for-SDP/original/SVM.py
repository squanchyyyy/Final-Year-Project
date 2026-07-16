import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import time 

start_time = time.time()

# 1️⃣ Load & Preprocess Data
data, meta = arff.loadarff('jm1_preprocessed.arff')
df = pd.DataFrame(data)
df.rename(columns={'defects': 'label'}, inplace=True)

# Convert binary labels from string to numeric
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(lambda x: x.decode(
        'utf-8') if isinstance(x, bytes) else x)
df['label'] = df['label'].map({'false': 0, 'true': 1})
df.dropna(inplace=True)

print("Data shape after processing:", df.shape)

# 2️⃣ Split Features & Labels
X = df.drop('label', axis=1)
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3️⃣ Scale Features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Train SVM Model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Generate Predictions & Evaluate
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)
y_pred_svm = y_pred

# Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = round(roc_auc_score(y_test, y_pred_proba), 4)

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
print(classification_report(y_test, y_pred, digits=4))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
"""Pseudocode
START

1. LOAD and PREPROCESS data
   - Load data from 'jm1.arff' file
   - Convert to DataFrame
   - Rename the 'defects' column to 'label'
   - Decode byte strings to readable text
   - Map 'false' → 0 and 'true' → 1 in 'label' column
   - Remove rows with missing values

2. SPLIT data into features and labels
   - X = all columns except 'label'
   - y = 'label' column

3. SPLIT dataset into training and test sets
   - Use 80% for training, 20% for testing
   - Set a random seed for reproducibility

4. SCALE the feature values
   - Use RobustScaler to normalize data and reduce impact of outliers
   - Fit on training data and transform both training and test sets

5. TRAIN a Support Vector Machine classifier
   - Use RBF kernel
   - Enable probability estimation
   - Fit the SVM model on the scaled training data

6. MAKE predictions
   - Predict class probabilities on the test data
   - Use threshold of 0.5 to convert probabilities into class labels

7. EVALUATE model performance
   - Compute precision, recall, F1-score for both classes
   - Compute ROC-AUC score
   - Print classification report

END
"""
