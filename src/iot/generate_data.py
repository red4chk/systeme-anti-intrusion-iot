import pandas as pd
import numpy as np
import random
import os

# Configuration
NUM_SAMPLES = 2000  # Nombre de lignes de données
OUTPUT_PATH = "data/iot/iot_data.csv"

def generate_iot_data(n=1000):
    print(f"🔄 Génération de {n} lignes de données simulées...")
    
    data = []
    
    for _ in range(n):
        # 1. Simulation de l'heure (0-23)
        hour = random.randint(0, 23)
        
        # Est-ce qu'il fait nuit ? (22h - 6h)
        is_night = (hour >= 22) or (hour <= 6)
        
        # Probabilité d'intrusion (plus faible en général, mais on veut assez d'exemples)
        # On force environ 30% d'intrusions pour que le modèle ait de quoi apprendre
        is_intrusion = random.random() < 0.3 
        
        if is_intrusion:
            # SCÉNARIO INTRUSION 🚨
            label = 1
            motion = 1  # Presque toujours du mouvement
            # Bruit élevé (ex: bris de glace, pas lourds) : entre 60 et 100
            sound_level = np.random.randint(60, 100) 
            # Vibration probable (bris de vitre/porte)
            vibration = 1 if random.random() > 0.2 else 0 
            # Température normale
            temperature = round(np.random.normal(20, 2), 1) 
            
        else:
            # SCÉNARIO NORMAL ✅
            label = 0
            
            # Mouvement : Rare la nuit, possible le jour
            if is_night:
                motion = 0 # Tout le monde dort
            else:
                motion = 1 if random.random() < 0.3 else 0 # Vie normale
            
            # Bruit : Faible en général
            sound_level = np.random.randint(20, 50)
            
            # Vibration : Aucune
            vibration = 0
            
            # Température normale
            temperature = round(np.random.normal(20, 2), 1)

        # Ajout à la liste
        data.append([motion, sound_level, vibration, temperature, hour, label])

    # Création du DataFrame
    columns = ['motion', 'sound_level', 'vibration', 'temperature', 'hour', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    return df

if __name__ == "__main__":
    # Génération
    df_iot = generate_iot_data(NUM_SAMPLES)
    
    # Vérification dossier
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Sauvegarde
    df_iot.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Fichier généré avec succès : {OUTPUT_PATH}")
    print("--- Aperçu des données ---")
    print(df_iot.head())
    print("\n--- Distribution des labels (0=Normal, 1=Intrusion) ---")
    print(df_iot['label'].value_counts())