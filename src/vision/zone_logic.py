import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
VIDEO_PATH = "data/videos/surveillance.mp4"
OUTPUT_PATH = "results/videos/output_zone.mp4"

def monitor_zone():
    print(f"🛡️  Surveillance de zone active sur : {VIDEO_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"❌ Vidéo introuvable.")

    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # --- DÉFINITION DE LA ZONE INTERDITE ---
    # Ici, je définis un polygone (carré) au centre-droit de l'image
    # Tu peux changer ces points selon ta vidéo !
    # Format : [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    zone_points = np.array([
    [0, height // 2],        # Haut gauche
    [width, height // 2],    # Haut droit
    [width, height],         # Bas droit
    [0, height]              # Bas gauche
], np.int32)

    
    zone_points = zone_points.reshape((-1, 1, 2))

    print("▶️ Analyse en cours... Une zone verte va s'afficher.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. Détection
        results = model(frame, conf=0.5, classes=0, verbose=False)
        
        # Copie de l'image pour dessiner dessus
        overlay = frame.copy()
        
        # Par défaut, pas d'intrusion
        intrusion_detected = False
        
        # 2. Vérifier chaque personne détectée
        for box in results[0].boxes:
            # Récupérer les coordonnées du cadre (bounding box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # On calcule le point central bas (les pieds de la personne)
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)
            
            # 3. Vérifier si les pieds sont DANS la zone
            # pointPolygonTest renvoie > 0 si c'est dedans
            result = cv2.pointPolygonTest(zone_points, (feet_x, feet_y), False)
            
            if result >= 0:
                intrusion_detected = True
                # Dessiner un cercle rouge aux pieds de l'intrus
                cv2.circle(frame, (feet_x, feet_y), 10, (0, 0, 255), -1)

        # 4. Gestion de l'affichage de la zone
        if intrusion_detected:
            color = (0, 0, 255) # ROUGE (Intrusion !)
            text = "!!! INTRUSION !!!"
        else:
            color = (0, 255, 0) # VERT (Sécurisé)
            text = "Zone Securisee"

        # Dessiner la zone semi-transparente
        cv2.fillPoly(overlay, [zone_points], color)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Dessiner les bords de la zone
        cv2.polylines(frame, [zone_points], True, color, 3)
        
        # Afficher le texte
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Afficher et sauvegarder
        cv2.imshow("Zone Logic", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Analyse terminée ! Vidéo sauvegardée dans : {OUTPUT_PATH}")

if __name__ == "__main__":
    monitor_zone()