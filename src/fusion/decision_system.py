import cv2
import numpy as np
import pandas as pd
import joblib
import random
from ultralytics import YOLO
import os

# Configuration
VIDEO_PATH = "data/videos/surveillance.mp4"
MODEL_IOT_PATH = "data/iot/model_iot.pkl"
SCALER_PATH = "data/iot/scaler.pkl"
OUTPUT_PATH = "results/videos/output_fusion_final.mp4"

def start_fusion_system():
    print("🧠 Démarrage du SYSTÈME DE FUSION (Affichage Optimisé)...")

    # 1. Chargement des modèles IA
    if not os.path.exists(MODEL_IOT_PATH) or not os.path.exists(SCALER_PATH):
         raise FileNotFoundError("❌ Modèles IoT manquants. Lancez d'abord la Partie A.")

    iot_model = joblib.load(MODEL_IOT_PATH)
    scaler = joblib.load(SCALER_PATH)
    vision_model = YOLO('yolov8n.pt')

    # 2. Préparation Vidéo
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError("❌ Vidéo introuvable.")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # --- NOUVELLE ZONE : TOUT LE BAS DE L'ÉCRAN ---
    mid_height = int(height / 2) # On coupe l'écran en deux
    zone_points = np.array([
        [0, mid_height],          # Milieu Gauche
        [width, mid_height],      # Milieu Droit
        [width, height],          # Bas Droit
        [0, height]               # Bas Gauche
    ], np.int32).reshape((-1, 1, 2))

    frame_count = 0
    current_iot_pred = 0 
    current_iot_data = []

    print("▶️ Système ACTIF. Zone interdite sur la moitié basse.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # --- PARTIE 1 : VISION (YOLO) ---
        video_intrusion = False
        results = vision_model(frame, conf=0.5, classes=0, verbose=False)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x, feet_y = int((x1 + x2) / 2), int(y2)
            
            # Test si les pieds sont dans la zone (moitié basse)
            if cv2.pointPolygonTest(zone_points, (feet_x, feet_y), False) >= 0:
                video_intrusion = True
                # Petit cadre rouge autour de la personne uniquement
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- PARTIE 2 : IOT (SIMULATION) ---
        if frame_count % 30 == 0:
            if random.random() < 0.2:
                sim_data = [[1, random.randint(70, 90), 1, 20.5, 23]] 
            else:
                sim_data = [[0, random.randint(30, 50), 0, 20.5, 14]]
            
            cols = ['motion', 'sound_level', 'vibration', 'temperature', 'hour']
            df_live = pd.DataFrame(sim_data, columns=cols)
            X_live = scaler.transform(df_live)
            current_iot_pred = iot_model.predict(X_live)[0]
            current_iot_data = sim_data[0]

        # --- PARTIE 3 : FUSION & AFFICHAGE OPTIMISÉ ---
        
        FINAL_ALERT = video_intrusion or (current_iot_pred == 1)
        status_color = (0, 0, 255) if FINAL_ALERT else (0, 255, 0) # Rouge ou Vert
        
        # 1. DESSIN DE LA ZONE (Discret)
        # Si intrusion vidéo : Zone rouge semi-transparente
        # Sinon : Ligne verte simple pour délimiter
        if video_intrusion:
             overlay = frame.copy()
             cv2.fillPoly(overlay, [zone_points], (0, 0, 255))
             cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        else:
             # Juste une ligne verte au milieu pour montrer la limite
             cv2.line(frame, (0, mid_height), (width, mid_height), (0, 255, 0), 2)

        # 2. BANDEAU D'INFORMATION (Plus petit)
        # Un bandeau de 80 pixels de haut seulement
        header_height = 80
        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        
        # Titre principal (Plus petit)
        status_text = "ALERTE INTRUSION" if FINAL_ALERT else "SECURISE"
        cv2.putText(frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Infos techniques (Petites et sur une ligne si possible)
        iot_msg = "ALERTE" if current_iot_pred == 1 else "OK"
        noise_val = current_iot_data[1] if current_iot_data else 0
        
        # Ligne 2 : Détails
        detail_text = f"[IoT: {iot_msg} (Bruit: {noise_val}dB)]  |  [Video: {'INTRUSION' if video_intrusion else 'OK'}]"
        cv2.putText(frame, detail_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("FUSION SYSTEM", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Terminé ! Vidéo sauvegardée : {OUTPUT_PATH}")

if __name__ == "__main__":
    start_fusion_system()