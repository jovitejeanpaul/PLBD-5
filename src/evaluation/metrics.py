import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def save_visual_report(y_test, y_pred, model_name):
    """Génère deux images : le rapport de classification et la matrice de confusion"""
    
    # --- 1. GÉNÉRATION DU RAPPORT (Precision, Recall, etc.) ---
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T 
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', cbar=False)
    plt.title(f'Performances détaillées - {model_name}')
    
    path_metrics = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../reports/figures/metrics_{model_name.lower()}.png'))
    plt.savefig(path_metrics)
    plt.close()

    # --- 2. GÉNÉRATION DE LA MATRICE DE CONFUSION ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # 'd' pour afficher les nombres entiers (Vrais Positifs, etc.)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédictions du modèle')
    plt.ylabel('Réalité (Vérité)')
    plt.title(f'Matrice de Confusion - {model_name}')
    
    path_cm = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../reports/figures/confusion_{model_name.lower()}.png'))
    plt.savefig(path_cm)
    plt.close()
    
    print(f"✅ Images sauvegardées pour {model_name} :")
    print(f"   - {os.path.basename(path_metrics)}")
    print(f"   - {os.path.basename(path_cm)}")