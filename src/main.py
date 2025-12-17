import sys
import os
import time

# On ajoute le dossier src au path pour que les imports fonctionnent bien
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import des modules que nous avons créés
from iot.generate_data import generate_iot_data
from iot.preprocess import load_and_preprocess_data
from iot.train_classifier import train_and_evaluate
from fusion.decision_system import start_fusion_system

def print_header():
    print("\n" + "="*50)
    print("🛡️  SYSTÈME ANTI-INTRUSION INTELLIGENT (SAII)  🛡️")
    print("="*50)

def full_setup():
    print("\n🔄 [ETAPE 1] INITIALISATION DU SYSTÈME...")
    print("------------------------------------------")
    
    # 1. Génération IoT
    print("1.1 Génération des données capteurs simulées...")
    df = generate_iot_data(2000)
    # Sauvegarde gérée dans generate_data, mais on assure le coup ici si besoin
    os.makedirs("data/iot", exist_ok=True)
    df.to_csv("data/iot/iot_data.csv", index=False)
    print("   -> Données sauvegardées.")
    time.sleep(1)

    # 2. Entraînement IA
    print("\n1.2 Entraînement du modèle de classification...")
    model, cm = train_and_evaluate()
    print("   -> Modèle Random Forest entraîné et sauvegardé.")
    time.sleep(1)

    print("\n✅ INITIALISATION TERMINÉE AVEC SUCCÈS.")

def launch_demo():
    print("\n🚀 [ETAPE 2] LANCEMENT DE LA DÉMO TEMPS RÉEL...")
    print("-----------------------------------------------")
    
    # Vérification des fichiers requis
    if not os.path.exists("data/iot/model_iot.pkl"):
        print("❌ Erreur : Modèle IA introuvable.")
        print("👉 Veuillez lancer l'option '1' (Installation) d'abord.")
        return

    if not os.path.exists("data/videos/surveillance.mp4"):
        print("❌ Erreur : Vidéo 'surveillance.mp4' introuvable dans data/videos/")
        return

    # Lancement du système de fusion
    try:
        start_fusion_system()
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt du système.")

def main():
    while True:
        print_header()
        print("1. 🛠️  INSTALLATION COMPLÈTE (Générer Data + Entraîner IA)")
        print("2. 👁️  LANCER LA DÉMO (Fusion IoT + Vidéo)")
        print("3. ❌  QUITTER")
        
        choice = input("\n👉 Votre choix (1-3) : ")

        if choice == '1':
            full_setup()
            input("\nAppuyez sur Entrée pour revenir au menu...")
        elif choice == '2':
            launch_demo()
            input("\nFin de la démo. Appuyez sur Entrée pour revenir au menu...")
        elif choice == '3':
            print("Fermeture du système. À bientôt !")
            break
        else:
            print("Choix invalide.")

if __name__ == "__main__":
    # S'assurer qu'on est à la racine du projet
    if not os.path.exists("src"):
        print("⚠️  ATTENTION : Veuillez lancer ce script depuis la racine du projet (anti_intrusion_project/)")
        print("Commande : python src/main.py")
    else:
        main()