import pandas as pd
import os
import joblib  # Pour sauvegarder le scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
INPUT_PATH = "data/iot/iot_data.csv"
SCALER_PATH = "data/iot/scaler.pkl"  # On sauvegarde l'outil de normalisation ici

def load_and_preprocess_data():
    print("🧹 Démarrage du prétraitement...")
    
    # 1. Chargement
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ Le fichier {INPUT_PATH} n'existe pas. Lance generate_data.py d'abord.")
    
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Vérification rapide
    if df.isnull().values.any():
        print("⚠️ Attention : Des valeurs manquantes ont été trouvées et supprimées.")
        df = df.dropna()

    # 3. Séparation Features (X) / Target (y)
    X = df.drop('label', axis=1)  # Tout sauf le label
    y = df['label']               # Juste le label (0 ou 1)
    
    # 4. Split Train / Test (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Normalisation (StandardScaler)
    # On calcule la moyenne/écart-type sur le TRAIN uniquement pour éviter la fuite de données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # On applique la même transformation sur le TEST
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Sauvegarde du Scaler pour utilisation future (Partie D - Fusion)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler sauvegardé dans {SCALER_PATH}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Test du script
    try:
        X_tr, X_te, y_tr, y_te = load_and_preprocess_data()
        print("\n✅ Prétraitement terminé avec succès !")
        print(f"📊 Données d'entraînement : {X_tr.shape} (Lignes, Colonnes)")
        print(f"📊 Données de test        : {X_te.shape} (Lignes, Colonnes)")
        print("\nExemple de ligne normalisée (Train[0]) :")
        print(X_tr[0])
    except Exception as e:
        print(f"❌ Erreur : {e}")