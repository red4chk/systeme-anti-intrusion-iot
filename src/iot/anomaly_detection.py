import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configuration
INPUT_PATH = "data/iot/iot_data.csv"
FIGURE_PATH = "results/figures/anomalies_detected.png"

def detect_anomalies():
    print("🕵️ Démarrage de la détection d'anomalies (Non supervisé)...")
    
    # 1. Chargement des données brutes
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"❌ {INPUT_PATH} introuvable.")
    
    df = pd.read_csv(INPUT_PATH)
    
    # On garde les données utiles (on enlève le label car l'algo ne doit pas tricher !)
    X = df[['motion', 'sound_level', 'vibration', 'temperature', 'hour']]
    
    # 2. Normalisation (Important pour que la température ne pèse pas moins que le bruit)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Isolation Forest
    # contamination=0.3 car on sait qu'on a généré environ 30% d'intrusions
    # Dans la vraie vie, on mettrait plutôt 0.01 ou 0.05 (événements rares)
    model = IsolationForest(contamination=0.3, random_state=42)
    model.fit(X_scaled)
    
    # 4. Prédiction
    # 1 = Normal, -1 = Anomalie
    predictions = model.predict(X_scaled)
    
    # On ajoute le résultat au DataFrame pour l'analyse
    df['anomaly_score'] = predictions
    
    # Combien d'anomalies trouvées ?
    n_anomalies = (predictions == -1).sum()
    print(f"🔎 Analyse terminée.")
    print(f"🔴 Anomalies détectées : {n_anomalies} sur {len(df)} enregistrements")
    
    return df

def visualize_anomalies(df):
    # On va visualiser : Niveau Sonore vs Heure
    # Les points rouges seront les anomalies détectées par l'algo
    
    plt.figure(figsize=(10, 6))
    
    # Points normaux (bleus)
    normal = df[df['anomaly_score'] == 1]
    plt.scatter(normal['hour'], normal['sound_level'], c='blue', alpha=0.5, label='Normal', s=20)
    
    # Anomalies (rouges)
    anomalies = df[df['anomaly_score'] == -1]
    plt.scatter(anomalies['hour'], anomalies['sound_level'], c='red', label='Anomalie détectée', marker='x', s=50)
    
    plt.title('Détection d\'Anomalies : Niveau Sonore selon l\'Heure')
    plt.xlabel('Heure de la journée (0-23h)')
    plt.ylabel('Niveau Sonore (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarde
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    plt.savefig(FIGURE_PATH)
    print(f"🖼️ Graphique sauvegardé : {FIGURE_PATH}")

if __name__ == "__main__":
    df_result = detect_anomalies()
    visualize_anomalies(df_result)