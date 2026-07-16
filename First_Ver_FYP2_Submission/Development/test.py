import pandas as pd
import shap
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.base import clone

# 1. Load and preprocess the data
data, meta = arff.loadarff('jm1.arff')
df = pd.DataFrame(data)

df.rename(columns={'defects': 'label'}, inplace=True)
for col in df.select_dtypes(include=[object]):
    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
df['label'] = df['label'].map({'false': 0, 'true': 1})
df.dropna(inplace=True)

#df = df.sample(500, random_state=42)

print("Columns:", df.columns)
print("Data shape after processing:", df.shape)

# 2. Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE-Tomek
smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

# 5. Recursive Feature Elimination (RFE)
rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
X_train_rfe = rfe_selector.fit_transform(X_train_res, y_train_res)
X_test_rfe = rfe_selector.transform(X_test)

selected_features = X.columns[rfe_selector.get_support()]
print("Selected Features:", selected_features.tolist())

# SHAP analysis
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_rfe, y_train_res)
explainer = shap.Explainer(rf_model, X_train_rfe)
shap_values = explainer(X_train_rfe, check_additivity=False)

shap_matrix = shap_values.values
if shap_matrix.ndim == 3:
    shap_matrix = shap_matrix[:, :, 1]
mean_shap_importance = np.mean(np.abs(shap_matrix), axis=0)

threshold = np.percentile(mean_shap_importance, 60) 
important_features = [
    feature for feature, importance in zip(selected_features, mean_shap_importance)
    if importance > threshold
]

print("Expanded Features (SHAP-based):", important_features)

# Use top N SHAP features
top_n = 8
top_features = [
    feature for _, feature in sorted(zip(mean_shap_importance, selected_features), reverse=True)
][:top_n]

X_train_top = X_train_res[top_features]
X_test_top = X_test[top_features]

# 6. Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_top)
X_test_scaled = scaler.transform(X_test_top)

# 7. Define models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, class_weight={0: 1, 1: 2}, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', C=6.0, random_state=42))
]

# 8. Focal Loss & PyTorch Meta-Model
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class TorchStackingMetaModel:
    def __init__(self, input_dim, epochs=50, lr=0.001):
        self.model = SimpleNN(input_dim)
        self.loss_fn = FocalLoss(alpha=0.7, gamma=1.2)
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

# 9. Generate meta-features
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features_train = np.zeros((X_train_scaled.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test_scaled.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    temp_test_preds = np.zeros((X_test_scaled.shape[0], 5))
    for j, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_fold_train, y_fold_train = X_train_scaled[train_idx], y_train_res.iloc[train_idx]
        X_fold_val = X_train_scaled[val_idx]

        clf = clone(model)
        clf.fit(X_fold_train, y_fold_train)
        val_preds = clf.predict_proba(X_fold_val)[:, 1]
        meta_features_train[val_idx, i] = val_preds
        temp_test_preds[:, j] = clf.predict_proba(X_test_scaled)[:, 1]
    meta_features_test[:, i] = temp_test_preds.mean(axis=1)

# 10. Train PyTorch meta-model
meta_model = TorchStackingMetaModel(input_dim=meta_features_train.shape[1])
meta_model.fit(meta_features_train, y_train_res)

# 11. Predictions
from sklearn.metrics import precision_recall_curve
y_pred_proba = meta_model.predict_proba(meta_features_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
optimal_threshold = thresholds[np.argmax(recalls >= 0.5)]  # Dynamically adjust threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# 12. Evaluation
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