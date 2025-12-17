import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hack pour importer preprocess.py qui est dans le même dossier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess_data

# Configuration
MODEL_PATH = "data/iot/model_iot.pkl"

def train_and_evaluate():
    print("🧠 Démarrage de l'entraînement du modèle IA...")

    # 1. Récupérer les données préparées
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. Initialiser le modèle (Random Forest)
    # C'est un excellent algo "passe-partout" robuste
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 3. Entraînement (Le moment où l'IA apprend)
    print("... Entraînement en cours ...")
    clf.fit(X_train, y_train)

    # 4. Prédiction sur les données de test (qu'elle n'a jamais vues)
    y_pred = clf.predict(X_test)

    # 5. Évaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Accuracy (Précision globale) : {acc * 100:.2f}%")
    print("\n📝 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # 6. Matrice de confusion (Optionnel : affichage console)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Vrai Négatif, Faux Positif, Faux Négatif, Vrai Positif):")
    print(cm)

    # 7. Sauvegarde du modèle entraîné
    joblib.dump(clf, MODEL_PATH)
    print(f"\n💾 Modèle entraîné sauvegardé dans : {MODEL_PATH}")
    
    return clf, cm

if __name__ == "__main__":
    model, cm = train_and_evaluate()
    
    # Bonus : Générer un graphique de la matrice de confusion si possible
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.title('Matrice de Confusion - Détection Intrusion')
        plt.savefig('results/figures/confusion_matrix_iot.png')
        print("🖼️ Graphique sauvegardé dans results/figures/confusion_matrix_iot.png")
    except Exception as e:
        print("Note : Graphique non généré (matplotlib/seaborn manquants ou erreur dossier).")