import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import each model's main function
from rf_model import main as rf_main
from svm_model import main as svm_main
from rf_svm_model import main as rf_svm_main
from rf_svm_ensemble import main as ensemble_main

# Run each model and collect metrics
metrics = {
    'Random Forest': rf_main(),
    'SVM': svm_main(),
    'RF + SVM': rf_svm_main(),
    'RF + SVM + GB': ensemble_main()
}

# Convert to DataFrame
df = pd.DataFrame(metrics).T  # Transpose for heatmap

# Plot Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
plt.title("Model Comparison Heatmap (Class 1 - Defective)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
