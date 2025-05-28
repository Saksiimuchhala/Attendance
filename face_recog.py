# using joblib to load face encodings and names, and KNN for recognition
import face_recognition
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from facenet_pytorch import MTCNN
import torch

# Initialize MTCNN once globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Load encodings and names
encodings, names = load("face_data.joblib")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(encodings, names)

def recognize_faces_in_frame(frame, knn_clf, threshold=0.5):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(rgb_frame)
    
    face_locations = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Convert (x1, y1, x2, y2) -> (top, right, bottom, left)
            top, right, bottom, left = y1, x2, y2, x1
            face_locations.append((top, right, bottom, left))
    else:
        face_locations = []
    
    # Get face encodings from detected boxes
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances, _ = knn_clf.kneighbors([face_encoding])
        confidence = (1 - distances[0][0]) * 100
        name = knn_clf.predict([face_encoding])[0]
        if confidence < threshold * 100:
            name = "Unknown"
        results.append((name, confidence))
        
        # Draw rectangle and label on frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({confidence:.1f}%)"
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1)
    
    return frame, results

# Webcam recognition
cap = cv2.VideoCapture(0)
print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated_frame, recognized = recognize_faces_in_frame(frame, knn)
    cv2.imshow("Face Recognition", annotated_frame)
    
    for name, confidence in recognized:
        print(f"Detected: {name} with confidence {confidence:.2f}%")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# using insightface buffalo_l model for face recognition
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.neighbors import KNeighborsClassifier

# # Load saved embeddings and labels
# data = np.load('face_data.npz')
# encodings = data['encodings']  # shape: (N, embedding_dim)
# names = data['names']          # shape: (N,)

# # Train KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
# knn.fit(encodings, names)

# # Initialize InsightFace model for live face detection and embedding extraction
# app = FaceAnalysis(name='buffalo_l')
# app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode

# # Start webcam
# cap = cv2.VideoCapture(0)  # 0 for default webcam

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Detect faces and get embeddings
#     faces = app.get(frame)

#     for face in faces:
#         bbox = face.bbox.astype(int)  # Bounding box: [x1, y1, x2, y2]
#         embedding = face.embedding.reshape(1, -1)

#         # Predict with KNN (recognize face)
#         pred_name = knn.predict(embedding)[0]
#         pred_dist, _ = knn.kneighbors(embedding, n_neighbors=1, return_distance=True)
#         confidence = 1 - pred_dist[0][0]  # similarity score (1 - cosine distance)

#         # Display box and name
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"{pred_name} ({confidence:.2f})"
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Show frame
#     cv2.imshow('Live Face Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
#         break

# cap.release()
# cv2.destroyAllWindows()





