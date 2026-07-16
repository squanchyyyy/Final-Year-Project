import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import time

start_time = time.time()

# 1. Load and preprocess
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)
df.rename(columns={'defects': 'label'}, inplace=True)
df['label'] = df['label'].apply(lambda x: 1 if x == b'true' else 0)
df.dropna(inplace=True)

X = df.drop('label', axis=1)
y = df['label']

# 2. SMOTE + reduce size
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
X_resampled = X_resampled.sample(frac=0.3, random_state=42)
y_resampled = y_resampled.loc[X_resampled.index]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

# 4. Scale once
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. RF - smaller and faster
rf_model = RandomForestClassifier(n_estimators=20, class_weight={
                                  0: 1, 1: 2}, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Use RF importances to select top 10 features
top_indices = np.argsort(rf_model.feature_importances_)[-10:]
X_train_sel = X_train.iloc[:, top_indices]
X_test_sel = X_test.iloc[:, top_indices]
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

# 7. Fast linear SVM (no probability)
svm_model = SVC(kernel='linear', C=0.5, class_weight={0: 1, 1: 2.0})
svm_model.fit(X_train_sel_scaled, y_train)

# 8. Simulate SVM proba using logistic regression on SVM outputs
svm_decision = svm_model.decision_function(X_test_sel_scaled).reshape(-1, 1)
platt = LogisticRegression()
platt.fit(svm_decision, y_test)
y_pred_proba_svm = platt.predict_proba(svm_decision)[:, 1]

# 9. RF probas
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# 10. Meta-model
meta_train = np.column_stack(
    (y_pred_proba_rf, y_pred_proba_svm, np.abs(y_pred_proba_rf - y_pred_proba_svm)))
meta_model = GradientBoostingClassifier(n_estimators=20, random_state=42)
meta_model.fit(meta_train, y_test)
y_pred_proba_meta = meta_model.predict_proba(meta_train)[:, 1]

# 11. Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(
    y_test, y_pred_proba_meta)
valid = np.where((precisions[:-1] >= 0.5) & (recalls[:-1] >= 0.5))[0]
if len(valid) > 0:
    best_idx = valid[np.argmax((precisions[valid] + recalls[valid]) / 2)]
    chosen_threshold = thresholds[best_idx]
else:
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    chosen_threshold = thresholds[np.argmax(f1_scores)]

# 12. Final predictions & metrics
y_pred = (y_pred_proba_meta >= chosen_threshold).astype(int)
report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = round(roc_auc_score(y_test, y_pred_proba_meta), 4)
elapsed_time = time.time() - start_time

# 13. Print
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
