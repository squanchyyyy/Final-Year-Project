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
data, meta = arff.loadarff('cm1.arff')
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
elapsed_time = time.time() - start_time

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
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")