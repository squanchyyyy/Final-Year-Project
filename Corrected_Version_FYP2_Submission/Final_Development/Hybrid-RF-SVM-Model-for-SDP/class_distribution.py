import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the ARFF dataset
def load_arff_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode byte strings to regular strings
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')

    # Identify and convert target column to binary (0/1)
    target_col = None
    if 'defects' in df.columns:
        target_col = 'defects'
    elif 'class' in df.columns:  # Some datasets use 'class' instead of 'defects'
        target_col = 'class'

    if target_col:
        df['defects'] = df[target_col].astype(str).str.lower().map({'false': 0, 'true': 1})
    else:
        raise ValueError("No target column ('defects' or 'class') found.")

    return df

# Visualize class distribution
def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    
    # Create countplot
    ax = sns.countplot(x='defects', data=df, palette='colorblind')
    plt.title('Defect Class Distribution', fontsize=16)
    plt.xlabel('Defect Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add percentage annotations
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 3,
                f'{height}\n({height/total:.1%})',
                ha='center', fontsize=11)
    
    # Customize plot appearance
    sns.despine()
    plt.xticks([0, 1], ['Non-Defective (0)', 'Defective (1)'])
    plt.tight_layout()
    plt.show()
    
    # Print class ratios
    class_counts = df['defects'].value_counts().sort_index()
    print("\nClass Distribution Summary:")
    if len(class_counts) == 2:
        print(f"Non-Defective: {class_counts[0]} samples ({class_counts[0]/total:.1%})")
        print(f"Defective: {class_counts[1]} samples ({class_counts[1]/total:.1%})")
        print(f"Imbalance Ratio: {class_counts[0]/class_counts[1]:.1f}:1")
    else:
        print("Warning: Only one class present in the dataset.")
        print(class_counts)

# Main execution
if __name__ == "__main__":
    # Load dataset
    file_path = 'jm1.arff'
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError
        
        df = load_arff_data(file_path)
        print("✅ Dataset loaded successfully!")
        print(f"📊 Total samples: {len(df)}")
        print(f"🧬 Features: {list(df.columns[:-1])}")  # Exclude target variable
        
        # Analyze class distribution
        plot_class_distribution(df)
        
        # Additional statistics
        print("\nAdditional Statistics:")
        print(df['defects'].describe())
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
    except ValueError as ve:
        print(f"❌ {str(ve)}")
    except Exception as e:
        print(f"⚠️ An unexpected error occurred: {str(e)}")
