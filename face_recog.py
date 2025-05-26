# # face_recognition_realtime.py
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
            if face is not None and face.size > 0:
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                print("No face detected in this frame.")

            resized_face = cv2.resize(rgb_face, (400, 400))  # Resize to match training format
            encodings = face_recognition.face_encodings(resized_face)

            if encodings:
                matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.7)
                face_distances = face_recognition.face_distance(known_encodings, encodings[0])
                best_match_index = np.argmin(face_distances)
            
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"
            

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
# import joblib
# from torchvision import transforms
# from torch.nn.functional import cosine_similarity

# # Load saved embeddings and names
# known_embeddings, known_names = joblib.load("face_embeddings.pkl")

# # Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Initialize MTCNN and FaceNet
# mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Transform
# preprocess = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

# # Video capture
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to PIL image
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     # Detect and align faces
#     boxes, faces = mtcnn.detect(img, landmarks=False)
#     faces_aligned = mtcnn(img)

#     if faces_aligned is not None:
#         for i, face_tensor in enumerate(faces_aligned):
#             if face_tensor is None:
#                 continue

#             # Get embedding
#             face_tensor = face_tensor.unsqueeze(0).to(device)
#             embedding = resnet(face_tensor)

#             # Compare with known embeddings
#             # Initialize
#             best_score = -1  # use -1 since cosine similarity ranges from -1 to 1
#             best_match_name = "Unknown"

#             # Compare embeddings and find best match
#             for known_embedding, known_name in zip(known_embeddings, known_names):
#                 known_embedding_tensor = torch.tensor(known_embedding).unsqueeze(0).to(device)
#                 similarity = cosine_similarity(embedding, known_embedding_tensor).item()
#                 if similarity > best_score:
#                     best_score = similarity
#                     best_match_name = known_name

#             # Apply threshold after comparison
#             if best_score >= 0.6:
#                 name = best_match_name
#             else:
#                 name = "Unknown"

#             # Draw box and name
#             x1, y1, x2, y2 = [int(b) for b in boxes[i]]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, name + f" ({best_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
