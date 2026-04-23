import joblib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Configuration du chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing import run_pipeline

def final_comparison_visual():
    print("📊 Génération du comparatif visuel...")
    
    # 1. Chargement et split des données
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/water_potability.csv'))
    X, y = run_pipeline(csv_path, return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models_to_compare = {
        "XGBoost": "../saved_models/xgboost_model.pkl",
        "SVM": "../saved_models/svm_model.pkl"
    }
    
    summary = []
    for name, relative_path in models_to_compare.items():
        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))
        if os.path.exists(full_path):
            model = joblib.load(full_path)
            y_pred = model.predict(X_test)
            
            summary.append({
                "Modèle": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 2),
                "Recall (Sécurité)": round(recall_score(y_test, y_pred), 2),
                "Precision": round(precision_score(y_test, y_pred), 2),
                "F1-Score": round(f1_score(y_test, y_pred), 2)
            })

    if not summary:
        print("❌ Erreur : Aucun modèle trouvé.")
        return

    # 2. Création du DataFrame
    df_res = pd.DataFrame(summary).set_index("Modèle")

    # 3. Génération de l'image (Heatmap)
    plt.figure(figsize=(10, 4))
    # On utilise une palette de couleurs (RdYlGn : Rouge-Jaune-Vert)
    sns.heatmap(df_res, annot=True, cmap='RdYlGn', fmt='.2f', cbar=True, linewidths=.5)
    
    plt.title('Comparaison Finale : XGBoost vs SVM (Aqua Sens)', fontsize=14, pad=20)
    plt.tight_layout()

    # Sauvegarde de l'image
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports/figures/model_comparison_results.png'))
    plt.savefig(img_path)
    plt.close()
    
    print(f"✅ Image comparative créée : reports/figures/model_comparison_results.png")
    print("\nTableau des scores :")
    print(df_res)

if __name__ == "__main__":
    final_comparison_visual()