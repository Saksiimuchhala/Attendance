import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import face_recognition
import cv2
import numpy as np

# Load known encodings and names
data = np.load("new.npz", allow_pickle=True)
known_encodings = data["encodings"]
known_names = data["names"]

def recognize_faces_in_frame(frame, known_encodings, known_names, threshold=50.0):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)


        if True in matches:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            confidence = (1 - best_distance) * 100

            if confidence > threshold:
                name = known_names[best_match_index]
            else:
                name = "Unknown"
                confidence = max(confidence, 0.0)  # Ensure confidence is non-negative
        else:
            name = "Unknown"
            confidence = 0.0

        results.append((name, confidence))

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({confidence:.1f}%)"
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (left + 6, top - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1)

    return frame, results

# Start webcam loop
cap = cv2.VideoCapture(0)
print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, recognized = recognize_faces_in_frame(frame, known_encodings, known_names)
    cv2.imshow("Face Recognition", annotated_frame)

    for name, confidence in recognized:
        print(f"Detected: {name} with confidence {confidence:.2f}%")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








# #using face_recognition , and KNN for recognition
# import face_recognition
# import cv2
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# # Load encodings and names from .npz
# data = np.load("new.npz", allow_pickle=True)
# encodings = data["encodings"]
# names = data["names"]

# # Train KNN
# knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
# knn.fit(encodings, names)

# def recognize_faces_in_frame(frame, knn_clf, threshold=0.4):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     face_locations = face_recognition.face_locations(rgb_frame)

#     # Get face encodings
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     results = []
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         distances, _ = knn_clf.kneighbors([face_encoding])
#         confidence = (1 - distances[0][0]) * 100
#         name = knn_clf.predict([face_encoding])[0]
#         if confidence < threshold * 100:
#             name = "Unknown"
#         results.append((name, confidence))

#         # Draw bounding box and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         label = f"{name} ({confidence:.1f}%)"
#         cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
#         cv2.putText(frame, label, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1)

#     return frame, results

# # Webcam loop
# cap = cv2.VideoCapture(0)
# print("Starting recognition. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated_frame, recognized = recognize_faces_in_frame(frame, knn)
#     cv2.imshow("Face Recognition", annotated_frame)

#     for name, confidence in recognized:
#         print(f"Detected: {name} with confidence {confidence:.2f}%")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()









# using cosine similarities 

# import numpy as np
# import cv2
# import face_recognition
# from sklearn.metrics.pairwise import cosine_similarity

# # Load stored face encodings
# data = np.load("new.npz", allow_pickle=True)
# known_encodings = data["encodings"]
# known_names = data["names"]

# def recognize_faces_in_frame(frame, known_encodings, known_names, threshold=0.5):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     results = []
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Compute cosine similarities
#         sims = cosine_similarity([face_encoding], known_encodings)[0]
#         best_idx = np.argmax(sims)
#         best_score = sims[best_idx]

#         name = known_names[best_idx] if best_score > threshold else "Unknown"
#         confidence = best_score * 100
#         results.append((name, confidence))

#         # Draw bounding box and label
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         label = f"{name} ({confidence:.1f}%)"
#         cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
#         cv2.putText(frame, label, (left + 6, top - 6),
#                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1)

#     return frame, results

# # Webcam loop
# cap = cv2.VideoCapture(0)
# print("Starting recognition. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated_frame, recognized = recognize_faces_in_frame(frame, known_encodings, known_names, threshold=0.5)
#     cv2.imshow("Face Recognition", annotated_frame)

#     for name, confidence in recognized:
#         print(f"Detected: {name} with confidence {confidence:.2f}%")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()











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

# # Initialize InsightFace model
# app = FaceAnalysis(name='buffalo_l')
# app.prepare(ctx_id=-1, det_size=(320, 320))  # Use CPU

# # Start webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     faces = app.get(frame)

#     for face in faces:
#         bbox = face.bbox.astype(int)
#         embedding = face.embedding.reshape(1, -1)

#         pred_name = knn.predict(embedding)[0]
#         pred_dist, _ = knn.kneighbors(embedding, n_neighbors=1, return_distance=True)
#         confidence = 1 - pred_dist[0][0]

#         # Label as Unknown if confidence is too low
#         if confidence < 0.50:
#             pred_name = "Unknown"

#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = f"{pred_name} ({confidence:.2f})"
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.imshow('Live Face Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
