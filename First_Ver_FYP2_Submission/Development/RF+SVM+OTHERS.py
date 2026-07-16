import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import shap
import time 

start_time = time.time()

# 1. Load & Preprocess Data
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)
df.rename(columns={'defects': 'label'}, inplace=True)

for col in df.select_dtypes(include=[object]):  
    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df['label'] = df['label'].map({'false': 0, 'true': 1})
df.dropna(inplace=True)
print("Data shape after processing:", df.shape)

# 2. Split Features & Labels
X = df.drop('label', axis=1)
y = df['label']

# 3. Apply SMOTE for balancing
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Scale Features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=50, class_weight={0: 1, 1: 2.0}, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 7. SHAP for Top 30 Important Features
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_scaled)
important_features = np.argsort(np.mean(np.abs(shap_values[1]), axis=0))[-30:]
X_train_selected = X_train.iloc[:, important_features]
X_test_selected = X_test.iloc[:, important_features]
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# 8. Calibrated SVM Model
svm_model = CalibratedClassifierCV(SVC(kernel='rbf', C=0.9, class_weight={0: 1, 1: 2.0}), cv=5)
svm_model.fit(X_train_selected_scaled, y_train)

# 9. Predict Probabilities
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_svm = svm_model.predict_proba(X_test_selected_scaled)[:, 1]

# 10. Train Meta-Model with Gradient Boosting and interaction feature
meta_train = np.column_stack((y_pred_proba_rf, y_pred_proba_svm, np.abs(y_pred_proba_rf - y_pred_proba_svm)))
meta_model = GradientBoostingClassifier(random_state=42)
meta_model.fit(meta_train, y_test)
y_pred_proba_meta = meta_model.predict_proba(meta_train)[:, 1]


# 11. Threshold Selection
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_meta)
valid = np.where((precisions[:-1] >= 0.5) & (recalls[:-1] >= 0.5))[0]
if len(valid) > 0:
    best_idx = valid[np.argmax((precisions[valid] + recalls[valid]) / 2)]
    chosen_threshold = thresholds[best_idx]
    print(f"Threshold with both ≥0.5 found: {chosen_threshold:.4f}")
else:
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    chosen_threshold = thresholds[np.argmax(f1_scores)]
    print(f"No threshold with both ≥0.5 found. Using best F1 threshold: {chosen_threshold:.4f}")

# 12. Final Predictions
y_pred = (y_pred_proba_meta >= chosen_threshold).astype(int)
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = round(roc_auc_score(y_test, y_pred_proba_meta), 4)

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

1. LOAD and PREPROCESS dataset
   - Read ARFF file into a DataFrame
   - Decode byte strings to regular strings
   - Convert 'label' column to binary (0, 1)
   - Drop any rows with missing values

2. SPLIT into features (X) and labels (y)

3. BALANCE dataset using SMOTE
   - Apply Synthetic Minority Oversampling to increase minority class

4. SPLIT balanced data into training and testing sets (e.g., 80/20)

5. SCALE features using RobustScaler
   - Fit and transform training features
   - Transform testing features

6. TRAIN Random Forest model
   - Use class weights to handle imbalance
   - Fit model on scaled training data

7. SELECT Top 30 important features using SHAP
   - Use SHAP values from Random Forest
   - Rank features by mean absolute SHAP value
   - Select top 30 features

8. RETRAIN scaled feature subset
   - Scale only the selected top 30 features (train and test)

9. TRAIN SVM with calibration
   - Use RBF kernel and class weights
   - Apply probability calibration using cross-validation
   - Fit on the selected scaled features

10. GET prediction probabilities from both base models
    - Random Forest → probabilities on full test data
    - SVM → probabilities on SHAP-selected test data

11. CREATE meta-features for stacking
    - Feature 1: RF probability
    - Feature 2: SVM probability
    - Feature 3: Absolute difference between RF and SVM probabilities

12. TRAIN meta-model using Gradient Boosting
    - Fit on meta-features and true test labels

13. GET final prediction probabilities from meta-model

14. SELECT classification threshold:
    - Use Precision-Recall curve
    - If any threshold gives both Precision ≥ 0.5 and Recall ≥ 0.5:
        - Choose threshold with highest (Precision + Recall) / 2
    - Else:
        - Use threshold with best F1-score

15. MAKE final predictions using selected threshold

16. EVALUATE model performance
    - Precision, Recall, F1-score for both classes
    - ROC-AUC score
    - Full classification report

17. PRINT execution time

END
"""