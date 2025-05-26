# face_recognition_realtime.py
import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np
import face_recognition
import time
import joblib

# Load saved embeddings
known_encodings, known_names = joblib.load("saved_encodings.pkl")
# print("Loaded known faces:", known_names)

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            name = "Unknown"
            if encodings:
                matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, encodings[0])
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 20):
        break

cap.release()
cv2.destroyAllWindows()
