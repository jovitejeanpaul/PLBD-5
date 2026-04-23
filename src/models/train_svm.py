import sys
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Configuration du chemin pour que Python trouve le dossier 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports de tes propres modules - CORRIGÉ ICI
from data_processing import run_pipeline
from evaluation.metrics import save_visual_report

def train_svm():
    print("🚀 Démarrage de l'entraînement du modèle SVM...")

    # 2. Chargement des données via le pipeline de ton ami
    # On précise le chemin du CSV et on demande le format (X, y)
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/water_potability.csv'))
    X, y = run_pipeline(csv_path, return_X_y=True)
    
    # 3. Découpage Entraînement / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("📊 Données chargées et découpées avec succès.")

    # 4. Définition du Pipeline avec normalisation
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            probability=True, 
            kernel='rbf', 
            C=1.0, 
            class_weight='balanced'  # Pour mieux détecter l'eau potable (classe 1)
        ))
    ])

    # 5. Entraînement
    print("🧠 Entraînement du SVM en cours...")
    model.fit(X_train, y_train)

    # 6. Prédictions et Évaluation Visuelle
    y_pred = model.predict(X_test)
    save_visual_report(y_test, y_pred, "SVM")

    # 7. Sauvegarde du modèle
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models'))
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"✅ Modèle sauvegardé dans : {model_path}")
    print("📈 Rapport visuel généré dans reports/figures/metrics_svm.png")

if __name__ == "__main__":
    train_svm()