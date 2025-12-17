import cv2
from ultralytics import YOLO
import os

# Configuration
VIDEO_PATH = "data/videos/surveillance.mp4"
OUTPUT_PATH = "results/videos/output_yolo.mp4"

def process_video():
    print(f"🎥 Chargement de la vidéo : {VIDEO_PATH}")
    
    # 1. Vérification du fichier
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"❌ Fichier vidéo introuvable : {VIDEO_PATH}. Merci d'ajouter une vidéo 'surveillance.mp4' dans data/videos/")

    # 2. Chargement du modèle YOLOv8 Nano (le plus léger et rapide)
    # Au premier lancement, il va le télécharger automatiquement depuis Internet.
    print("🚀 Chargement du modèle YOLOv8n...")
    model = YOLO('yolov8n.pt') 

    # 3. Ouverture de la vidéo
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Récupération des propriétés de la vidéo pour la sauvegarde
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Préparation de l'écriture vidéo
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("▶️ Début du traitement frame par frame... (Appuie sur 'q' pour quitter la fenêtre)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. Détection avec YOLO
        # classes=0 signifie qu'on ne garde que la classe "Personne"
        # conf=0.5 signifie qu'il faut être sûr à 50% minimum
        results = model.predict(frame, conf=0.5, classes=0, verbose=False)

        # 5. Dessiner les résultats sur l'image
        annotated_frame = results[0].plot()

        # Affichage en direct (optionnel, peut être lent sur certains PC)
        cv2.imshow("YOLOv8 Detection - Système Anti-Intrusion", annotated_frame)

        # Sauvegarde de la frame
        out.write(annotated_frame)

        # Quitter si on appuie sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nettoyage
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Traitement terminé ! Vidéo sauvegardée dans : {OUTPUT_PATH}")

if __name__ == "__main__":
    try:
        process_video()
    except Exception as e:
        print(f"❌ Erreur : {e}")