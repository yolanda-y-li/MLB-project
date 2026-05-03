import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set academic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Define the path to the results folder
RESULTS_DIR = os.path.join('tests', 'results')

def generate_figures():
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Could not find directory {RESULTS_DIR}.")
        return

    # 1. Confusion Matrix Heatmap
    try:
        path = os.path.join(RESULTS_DIR, 'confusion_matrix.csv')
        cm_df = pd.read_csv(path)
        test_cm = cm_df[(cm_df['split'] == 'test') & (cm_df['normalized'] == 1)]
        pivot_cm = test_cm.pivot(index='true_class', columns='pred_class', values='count')
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_cm, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Normalized Confusion Matrix (Test Set)')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig('figure1_confusion_matrix.png', dpi=300)
        print("Generated Figure 1: figure1_confusion_matrix.png")
    except Exception as e:
        print(f"Error generating Fig 1: {e}")

    # 2. Per-Class F1 Scores
    try:
        path = os.path.join(RESULTS_DIR, 'per_class_metrics.csv')
        class_df = pd.read_csv(path)
        test_metrics = class_df[(class_df['split'] == 'test') & (class_df['class_id'] != -1)]
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='class_name', y='f1', data=test_metrics, palette='viridis')
        plt.ylim(0, 1)
        plt.title('Model Performance (F1-Score) by Interaction Type')
        plt.ylabel('F1-Score')
        plt.xlabel('Interaction Class')
        plt.tight_layout()
        plt.savefig('figure2_class_performance.png', dpi=300)
        print("Generated Figure 2: figure2_class_performance.png")
    except Exception as e:
        print(f"Error generating Fig 2: {e}")

    # 3. Cold Start Performance (Generalization Gap)
    try:
        path = os.path.join(RESULTS_DIR, 'cold_start.csv')
        cold_df = pd.read_csv(path)
        test_cold = cold_df[cold_df['split'] == 'test']
        
        plt.figure(figsize=(9, 6))
        sns.barplot(x='bucket', y='auroc', data=test_cold, palette='magma')
        plt.ylim(0.4, 1.0) 
        plt.axhline(0.5, ls='--', color='red', alpha=0.5, label='Random Guessing')
        plt.title('Generalization: Performance on Unseen Entities')
        plt.ylabel('AUROC')
        plt.xlabel('Data Split Bucket')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure3_cold_start.png', dpi=300)
        print("Generated Figure 3: figure3_cold_start.png")
    except Exception as e:
        print(f"Error generating Fig 3: {e}")

    # 4. Ranking Hits@K Curve
    try:
        path = os.path.join(RESULTS_DIR, 'ranking_metrics.csv')
        rank_df = pd.read_csv(path)
        hits_data = rank_df[rank_df['metric'].str.contains('hits@')].copy()
        hits_data['k'] = hits_data['metric'].str.extract(r'(\d+)').astype(int)
        hits_data = hits_data.sort_values('k')

        plt.figure(figsize=(8, 6))
        plt.plot(hits_data['k'], hits_data['value'], marker='o', linestyle='-', color='teal', linewidth=2)
        plt.fill_between(hits_data['k'], hits_data['value'], color='teal', alpha=0.1)
        plt.title('Search Utility: Hits@K Curve')
        plt.xlabel('Top-K Candidates')
        plt.ylabel('Probability of Finding True Interaction')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('figure4_hits_at_k.png', dpi=300)
        print("Generated Figure 4: figure4_hits_at_k.png")
    except Exception as e:
        print(f"Error generating Fig 4: {e}")

if __name__ == "__main__":
    generate_figures()