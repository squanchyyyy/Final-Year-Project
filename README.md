# Research-Based Final Year Project - Applying Machine Learning Techniques to Predict Software Defects 👾🧟‍♀️

# Project Overview
Software Defect Prediction (SDP) is crucial for improving software reliability and minimizing maintenance expenses by pinpointing faulty modules prior to deployment. Traditional single machine learning models frequently suffer from challenges like class imbalance, noisy metrics, and high computational complexity.  This project introduces an innovative hybrid machine learning framework—RF-SVM+—which resolves these issues by integrating the feature-ranking capabilities of Random Forest (RF), the high-speed classification performance of a linear Support Vector Machine (SVM), and a Gradient Boosting meta-classifier to deliver highly accurate and robust final predictions. 

## Key Challenges Addressed
- **Class Imbalance**: Mitigated using Synthetic Minority Over-sampling Technique (SMOTE) alongside class-weight configurations.  

- **Computational Complexity**: Handled via rapid feature reduction using RF and an optimized linear SVM to drastically reduce training and execution times.

## Framework Architecture
The proposed RF-SVM+ framework operates in a unified, multi-stage pipeline:
<img width="1143" height="866" alt="image" src="https://github.com/user-attachments/assets/1670723d-91a1-48fe-90d6-b77a24a4a3f0" />
1. **Data Preprocessing & Balancing**: Decodes byte-string metrics, handles missing values, and balances minority defect classes using SMOTE. Data is standardized using RobustScaler to limit outlier impact.
2. **Feature Selection**: Random Forest evaluates and ranks feature importance scores to lower dimensionality.
3. **Base Classification & Calibration**: A fast linear SVM handles primary classification, augmented with Platt Scaling to output well-calibrated probabilities.
4. **Meta-Learning Ensemble**: A Gradient Boosting meta-model is trained on the probabilistic outputs and interaction terms of the base models to yield the ultimate prediction.  

## Dataset & Metrics
Experiments were carried out using the benchmark NASA JM1 dataset from the PROMISE repository.

**Dataset Details**
- **Total Instances**: 10,885 software modules.  
- **Language Base**: C programming language ground systems.  
- **Class Distribution**: 80.65% non-defective modules (Class 0) and 19.35% defective modules (Class 1).
- **Feature Count**: 22 software code metrics.  

**Metric Categories Evaluated**
- **McCabe Metrics**: Cyclomatic complexity ($V(G)$), essential complexity ($ev(G)$), design complexity ($iv(G)$), and total lines of code ($loc$).
- **Halstead Metrics**: Unique operators/operands, total operators/operands, volume ($v$), difficulty ($d$), effort ($e$), time ($t$), and estimated bugs ($b$).
- **Line-Based & Flow Metrics**: Lines of code (IOCode), blank lines (IOBlank), comment lines (IOComment), and total branch count (branchCount).

## Experimental Results 
The enhanced hybrid RF-SVM+ model substantially outperformed individual baseline classifiers.

<img width="407" height="383" alt="image" src="https://github.com/user-attachments/assets/37e4aaf4-4845-456f-a37f-42f38ede016f" />

RF-SVM+ having extremely low run time 🤟

<img width="527" height="407" alt="image" src="https://github.com/user-attachments/assets/0a16d9b6-8be1-4c43-b2f4-33cabc8ce514" />

The hybrid system successfully handles highly skewed datasets, achieving an exceptional F1-score on the minority class while keeping the processing footprint lightweight enough for automated CI/CD software quality assurance pipelines.

# Let's get started !!!
**Prerequisites**
- Python 3.8+
- Visual Studio or any preferred IDE

**Installation & Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/squanchyyyy/Final-Year-Project.git
   cd Final-Year-Project
2. Navigate to the final version directory:
   The repository contains two versions of the project (first ver and corrected ver). The corrected    ver directory represents the final, fully optimized implementation:
   ```bash
   cd "Corrected_Version_FYP2_Submission/Final Development/Hybrid-RF-SVM-Model-for-SDP"
3. Install the necessary dependencies:
   ```bash
   pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn
4. Run the implementation pipeline:
   To train and evaluate the final proposed hybrid framework, run the main model script:
   ```bash
   python RF-SVM+.py

**📊 Supplementary Scripts**
If you want to run individual components of the research (such as data analysis or baseline models), you can execute them separately:
- **Data Analysis & Exploratory Visualization**:
  - Run `summary_statistic.py` for dataset summaries.
  - Run `class_distribution.py` to view data balance charts.
  - Run `correlation_matrix.py` or `correlation_with_target.py` for feature relationship insights.
- **Baseline Model Evaluations**:
  - Run `RF.py` to test the standalone Random Forest model.
  - Run `SVM.py` to test the standalone Support Vector Machine model.
  - Run `RF-SVM.py` to test the standard hybrid approach without meta-optimization.
   
## Acknowledgements 🎀
- Author: Lew Eng Jean
- Supervisor: Ms. Venushini A/P Rajendran
- Institution: Faculty of Computing & Informatics, Multimedia University (MMU)


