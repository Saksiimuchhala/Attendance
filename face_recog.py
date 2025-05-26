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
















# face_recognition using compare_faces of face_recognition library
# import cv2
# from facenet_pytorch import MTCNN
# import torch
# from PIL import Image
# import numpy as np
# import face_recognition
# import time
# import joblib

# # Load saved embeddings
# known_encodings, known_names = joblib.load("saved_encodings.pkl")
# # print("Loaded known faces:", known_names)

# # Initialize MTCNN for face detection
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)

# cap = cv2.VideoCapture(0)
# start_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     boxes, _ = mtcnn.detect(img)

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = [int(b) for b in box]
#             face = frame[y1:y2, x1:x2]
#             if face is not None and face.size > 0:
#                 rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             else:
#                 print("No face detected in this frame.")

#             resized_face = cv2.resize(rgb_face, (400, 400))  # Resize to match training format
#             encodings = face_recognition.face_encodings(resized_face)

#             if encodings:
#                 matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.7)
#                 face_distances = face_recognition.face_distance(known_encodings, encodings[0])
#                 best_match_index = np.argmin(face_distances)
            
#                 if matches[best_match_index]:
#                     name = known_names[best_match_index]
#                 else:
#                     name = "Unknown"
#             else:
#                 name = "Unknown"
            

#             # Draw bounding box and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
#         break

# cap.release()
# cv2.destroyAllWindows()






# using KNN for face recognition
# import cv2
# from facenet_pytorch import MTCNN
# import torch
# from PIL import Image
# import face_recognition
# import time
# import joblib
# from sklearn.neighbors import KNeighborsClassifier

# # Load saved embeddings and names
# known_encodings, known_names = joblib.load("saved_encodings.pkl")

# # Train KNN classifier
# knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
# knn.fit(known_encodings, known_names)

# # Initialize MTCNN for face detection
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)

# cap = cv2.VideoCapture(0)
# start_time = time.time()

# print("Press 'q' to quit or wait 30 seconds.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     boxes, _ = mtcnn.detect(img)

#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = [int(b) for b in box]
#             face = frame[y1:y2, x1:x2]

#             if face is None or face.size == 0:
#                 continue

#             rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             resized_face = cv2.resize(rgb_face, (400, 400))  # match training size
#             encodings = face_recognition.face_encodings(resized_face)

#             if not encodings:
#                 name_to_show = "Unknown"
#                 confidence = 0
#             else:
#                 encoding = encodings[0]
#                 distances, _ = knn.kneighbors([encoding])
#                 closest_distance = distances[0][0]
#                 predicted_name = knn.predict([encoding])[0]
#                 confidence = (1 - closest_distance) * 100

#                 if confidence >= 50:
#                     name_to_show = predicted_name
#                 elif confidence < 20:
#                     name_to_show = "Unknown"
#                 else:
#                     name_to_show = predicted_name

#             # Draw box and label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, name_to_show, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
#         break

# cap.release()
# cv2.destroyAllWindows()






# import cv2
# from facenet_pytorch import MTCNN
# import torch
# from PIL import Image
# import numpy as np
# import face_recognition
# import time
# import joblib
# import os
# from collections import defaultdict

# class LiveFaceRecognizer:
#     def __init__(self):
#         self.load_encodings()
#         self.setup_detection()
#         self.setup_tracking()
        
#     def load_encodings(self):
#         """Load encodings with enhanced data if available"""
#         try:
#             if os.path.exists("enhanced_encodings.pkl"):
#                 data = joblib.load("enhanced_encodings.pkl")
#                 self.known_encodings = data['encodings']
#                 self.known_names = data['names']
#                 self.person_encodings = data.get('person_encodings', {})
#                 tolerance_info = data.get('tolerance_info', {})
#                 self.tolerance = tolerance_info.get('suggested_tolerance', 0.6)
#                 print(f"Loaded enhanced encodings: {len(set(self.known_names))} people")
#                 print(f"Using tolerance: {self.tolerance:.3f}")
#             else:
#                 self.known_encodings, self.known_names = joblib.load("saved_encodings.pkl")
#                 self.person_encodings = {}
#                 self.tolerance = 0.6
#                 print(f"Loaded basic encodings: {len(set(self.known_names))} people")
#                 print("Using default tolerance: 0.6")
#         except Exception as e:
#             print(f"Error loading encodings: {e}")
#             raise
            
#     def setup_detection(self):
#         """Setup MTCNN optimized for live video"""
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.mtcnn = MTCNN(
#             keep_all=True,
#             device=self.device,
#             min_face_size=50,  # Reasonable minimum for live video
#             thresholds=[0.8, 0.85, 0.9],  # High thresholds for speed
#             factor=0.85,  # Less aggressive scaling
#             post_process=False  # Disable for speed
#         )
        
#         # Pre-compute common sizes for faster processing
#         self.detection_size = (320, 240)  # Small size for detection
        
#     def setup_tracking(self):
#         """Setup lightweight tracking for live video"""
#         self.last_results = []
#         self.frame_skip_counter = 0
#         self.skip_frames = 2  # Process every 3rd frame for recognition, but detect every frame
#         self.permanent_labels = {}  # Store permanent labels for faces
#         self.face_id_counter = 0
        
#     def detect_faces_fast(self, frame):
#         """Fast face detection on every frame"""
#         # Resize for detection
#         small_frame = cv2.resize(frame, self.detection_size)
        
#         # Convert to PIL
#         img_pil = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        
#         # Detect faces
#         try:
#             boxes, probs = self.mtcnn.detect(img_pil)
#         except:
#             return []
        
#         face_locations = []
#         if boxes is not None:
#             h_scale = frame.shape[0] / self.detection_size[1]
#             w_scale = frame.shape[1] / self.detection_size[0]
            
#             for box, prob in zip(boxes, probs):
#                 if prob > 0.85:  # High confidence only
#                     # Scale back to original size
#                     x1, y1, x2, y2 = box
#                     x1 = int(x1 * w_scale)
#                     y1 = int(y1 * h_scale)
#                     x2 = int(x2 * w_scale)
#                     y2 = int(y2 * h_scale)
                    
#                     # Bounds check
#                     h, w = frame.shape[:2]
#                     x1, y1 = max(0, x1), max(0, y1)
#                     x2, y2 = min(w, x2), min(h, y2)
                    
#                     if x2 > x1 and y2 > y1:
#                         face_locations.append((x1, y1, x2, y2))
        
#         return face_locations
    
#     def recognize_face_fast(self, face_image):
#         """Fast face recognition"""
#         try:
#             # Quick resize
#             face_resized = cv2.resize(face_image, (150, 150))
            
#             # Get encoding with minimal jitter
#             encodings = face_recognition.face_encodings(face_resized, num_jitters=1)
#             if not encodings:
#                 return "Unknown", 0.0
            
#             face_encoding = encodings[0]
            
#             # Fast comparison
#             if self.person_encodings:
#                 best_distance = float('inf')
#                 best_name = "Unknown"
                
#                 for person, person_encs in self.person_encodings.items():
#                     distances = face_recognition.face_distance(person_encs, face_encoding)
#                     min_dist = np.min(distances)
#                     if min_dist < best_distance:
#                         best_distance = min_dist
#                         best_name = person
#             else:
#                 distances = face_recognition.face_distance(self.known_encodings, face_encoding)
#                 best_match_idx = np.argmin(distances)
#                 best_distance = distances[best_match_idx]
#                 best_name = self.known_names[best_match_idx]
            
#             # Calculate confidence
#             confidence = max(0, (1 - best_distance) * 100)
            
#             # Apply tolerance
#             if best_distance <= self.tolerance:
#                 return best_name, confidence
#             else:
#                 return "Unknown", confidence
                
#         except Exception as e:
#             return "Unknown", 0.0
    
#     def get_face_id(self, x1, y1, x2, y2):
#         """Get or assign face ID based on position"""
#         face_center_x = (x1 + x2) // 2
#         face_center_y = (y1 + y2) // 2
        
#         # Check if this face matches any existing permanent labels
#         for face_id, data in self.permanent_labels.items():
#             stored_center_x, stored_center_y = data['center']
#             # If face is within 80 pixels of a known face, it's the same person
#             if abs(face_center_x - stored_center_x) < 80 and abs(face_center_y - stored_center_y) < 80:
#                 # Update position
#                 self.permanent_labels[face_id]['center'] = (face_center_x, face_center_y)
#                 return face_id
        
#         # New face - assign new ID
#         self.face_id_counter += 1
#         face_id = self.face_id_counter
#         self.permanent_labels[face_id] = {
#             'center': (face_center_x, face_center_y),
#             'name': None,
#             'confidence': 0.0,
#             'last_seen': 0
#         }
#         return face_id

#     def process_live_frame(self, frame):
#         """Process live camera frame efficiently with permanent labels"""
#         # Always detect faces for smooth tracking
#         face_locations = self.detect_faces_fast(frame)
        
#         # Only do recognition every few frames
#         self.frame_skip_counter += 1
#         do_recognition = (self.frame_skip_counter % (self.skip_frames + 1) == 0)
        
#         results = []
#         current_frame_faces = set()
        
#         for x1, y1, x2, y2 in face_locations:
#             # Extract face
#             face = frame[y1:y2, x1:x2]
            
#             if face.size == 0:
#                 continue
            
#             # Get or assign face ID
#             face_id = self.get_face_id(x1, y1, x2, y2)
#             current_frame_faces.add(face_id)
            
#             # Update last seen
#             self.permanent_labels[face_id]['last_seen'] = self.frame_skip_counter
            
#             # Check if we already have a permanent label for this face
#             if self.permanent_labels[face_id]['name'] is not None:
#                 # Use permanent label
#                 name = self.permanent_labels[face_id]['name']
#                 confidence = self.permanent_labels[face_id]['confidence']
#             else:
#                 # Need to do recognition
#                 if do_recognition:
#                     # Convert to RGB
#                     rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                     # Do recognition
#                     name, confidence = self.recognize_face_fast(rgb_face)
                    
#                     # If we got a good recognition result, make it permanent
#                     if name != "Unknown" and confidence > 40:
#                         self.permanent_labels[face_id]['name'] = name
#                         self.permanent_labels[face_id]['confidence'] = confidence
#                         print(f"Permanently labeled face {face_id} as {name} ({confidence:.1f}%)")
#                     else:
#                         # For unknown faces, keep checking
#                         name = "Unknown"
#                         confidence = 0.0
#                 else:
#                     # Use temporary "Unknown" until next recognition
#                     name = "Unknown"
#                     confidence = 0.0
            
#             results.append({
#                 'box': (x1, y1, x2, y2),
#                 'name': name,
#                 'confidence': confidence,
#                 'face_id': face_id
#             })
        
#         # Clean up old faces (not seen for 100 frames)
#         faces_to_remove = []
#         for face_id, data in self.permanent_labels.items():
#             if self.frame_skip_counter - data['last_seen'] > 100:
#                 faces_to_remove.append(face_id)
        
#         for face_id in faces_to_remove:
#             print(f"Removing old face {face_id}: {self.permanent_labels[face_id]['name']}")
#             del self.permanent_labels[face_id]
        
#         return results

# def main():
#     try:
#         recognizer = LiveFaceRecognizer()
#     except Exception as e:
#         print(f"Failed to initialize recognizer: {e}")
#         return
    
#     # Initialize camera optimized for live video
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return
    
#     print("Starting live face recognition...")
#     print("Press 'q' to quit")
    
#     # FPS calculation
#     frame_count = 0
#     fps_start_time = time.time()
#     fps = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read frame")
#             break
        
#         frame_count += 1
        
#         # Process frame (every frame, but recognition is optimized)
#         start_time = time.time()
#         results = recognizer.process_live_frame(frame)
#         process_time = time.time() - start_time
        
#         # Draw results - green boxes for recognized faces, red boxes for unknown faces
#         for result in results:
#             x1, y1, x2, y2 = result['box']
#             name = result['name']
#             confidence = result['confidence']
#             face_id = result['face_id']
            
#             if name != "Unknown":
#                 # Green box for recognized faces
#                 color = (0, 255, 0)  # Green
#                 label = f"{name} ({confidence:.1f}%)"
#             else:
#                 # Red box for unknown faces
#                 color = (0, 0, 255)  # Red
#                 label = f"Unknown (ID: {face_id})"
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
#             # Draw label with background
#             label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#             cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
#                          (x1 + label_size[0], y1), color, -1)
#             cv2.putText(frame, label, (x1, y1 - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Calculate and display FPS
#         if frame_count % 30 == 0:
#             elapsed = time.time() - fps_start_time
#             fps = 30 / elapsed if elapsed > 0 else 0
#             fps_start_time = time.time()
        
#         # Display stats
#         cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, f"Process: {process_time*1000:.1f}ms", (10, 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         cv2.putText(frame, f"Faces: {len(results)}", (10, 85), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         cv2.putText(frame, f"Permanent Labels: {len([f for f in recognizer.permanent_labels.values() if f['name'] is not None])}", (10, 110), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
#         # Show frame
#         cv2.imshow("Live Face Recognition", frame)
        
#         # Handle key press
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Live recognition stopped")

# if __name__ == "__main__":
#     main()