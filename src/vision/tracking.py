import cv2
from ultralytics import YOLO
import os

# Configuration
VIDEO_PATH = "data/videos/surveillance.mp4"
OUTPUT_PATH = "results/videos/output_tracking.mp4"

def track_objects():
    print(f"🕵️  Démarrage du Tracking sur : {VIDEO_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"❌ Vidéo introuvable.")

    # Chargement du modèle
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("▶️ Tracking en cours... (Regarde les numéros au-dessus des têtes)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 🔧 C'EST ICI QUE LA MAGIE OPÈRE : persist=True
        # Cela active le tracking (l'IA se "souvient" des images précédentes)
        results = model.track(frame, persist=True, conf=0.5, classes=0, verbose=False)

        # Récupération de l'image annotée par YOLO
        annotated_frame = results[0].plot()

        # Bonus : On peut récupérer les IDs manuellement si on veut faire des stats
        if results[0].boxes.id is not None:
            # Récupère les IDs uniques présents sur l'image
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # On pourrait afficher ici : "Personnes détectées : ID 1, ID 2..."

        cv2.imshow("Tracking - IDs Uniques", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Tracking terminé ! Vidéo sauvegardée dans : {OUTPUT_PATH}")

if __name__ == "__main__":
    track_objects()