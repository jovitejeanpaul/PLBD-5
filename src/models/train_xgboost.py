import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import save_visual_report
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Astuce pour pouvoir importer data_processing qui est dans le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing import run_pipeline

def train_xgboost():
    print("🚀 Démarrage de l'entraînement XGBoost...")
    
    # 1. Récupération des données via le pipeline de ton ami
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/water_potability.csv'))
    X, y = run_pipeline(csv_path, return_X_y=True)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Modèle
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 4. Sauvegarde du modèle (pour ne pas avoir à le relancer)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models/xgboost_model.pkl'))
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé dans : {model_path}")
    
    # 5. ÉVALUATION ET SAUVEGARDE DES RAPPORTS
    y_pred = model.predict(X_test)
    save_visual_report(y_test,y_pred,"XGBoost")
    
    # Rapport texte
    report = classification_report(y_test, y_pred)
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports/logs/xgboost_report.txt'))
    with open(log_path, 'w') as f:
        f.write(report)
    
    # Matrice de confusion (Image)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion - XGBoost')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    
    fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports/figures/confusion_xgboost.png'))
    plt.savefig(fig_path)
    plt.close()
    
    print(f"📊 Statistiques et images sauvegardées dans le dossier /reports/")

if __name__ == "__main__":
    train_xgboost()