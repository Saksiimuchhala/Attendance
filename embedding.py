# # create encodings using face_recognition 
# import os
# import face_recognition
# import numpy as np

# def load_training_data(train_dir):
#     face_encodings = []
#     face_names = []

#     for person_name in os.listdir(train_dir):
#         person_dir = os.path.join(train_dir, person_name)
#         if os.path.isdir(person_dir):
#             for img_name in os.listdir(person_dir):
#                 img_path = os.path.join(person_dir, img_name)
#                 image = face_recognition.load_image_file(img_path)
#                 face_locations = face_recognition.face_locations(image)
#                 encodings = face_recognition.face_encodings(image, face_locations)
#                 if len(encodings) > 0:
#                     face_encodings.append(encodings[0])
#                     face_names.append(person_name)
#                 else:
#                     print(f"No face encoding found in {img_path}")

#     return np.array(face_encodings), np.array(face_names)

# # Load and save
# train_dir = "1_Data"
# encodings, names = load_training_data(train_dir)
# np.savez_compressed("new.npz", encodings=encodings, names=names)

# print(f"Saved {len(encodings)} face encodings ")




#face embeddings of single images  

# import os
# import cv2
# import face_recognition
# import numpy as np

# def load_training_data(train_dir):
#     face_encodings = []
#     face_names = []

#     for img_name in os.listdir(train_dir):
#         img_path = os.path.join(train_dir, img_name)
#         if os.path.isfile(img_path):
#             person_name = os.path.splitext(img_name)[0]
            

#             image = face_recognition.load_image_file(img_path)
#             face_locations = face_recognition.face_locations(image)
#             encodings = face_recognition.face_encodings(image, face_locations)

#             # Convert to BGR for OpenCV display
#             image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             if len(face_locations) > 0:
#                 for (top, right, bottom, left) in face_locations:
#                     cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
#                     cv2.putText(image_bgr, person_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 cv2.imshow("Processing", image_bgr)
#                 cv2.waitKey(300)  # Wait 300 ms before processing next image

#                 face_encodings.append(encodings[0])
#                 face_names.append(person_name)
#             else:
#                 print(f"No face encoding found in {img_path}")

#     cv2.destroyAllWindows()
#     return np.array(face_encodings), np.array(face_names)

# # Load and save
# train_dir = "Data_1"
# encodings, names = load_training_data(train_dir)
# np.savez_compressed("new_1.npz", encodings=encodings, names=names)

# print(f"Saved {len(encodings)} face encodings ")



import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime

def load_training_data(train_dir):
    face_encodings = []
    face_names = []

    app = FaceAnalysis(name='buffalo_l')  # MobileFaceNet model bundle
    app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode

    for person_name in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                faces = app.get(img)
                if len(faces) > 0:
                    embedding = faces[0].embedding
                    face_encodings.append(embedding)
                    face_names.append(person_name)
                else:
                    print(f"No face detected in {img_path}")

    return np.array(face_encodings), np.array(face_names)

# Load data and save as .npz
train_dir = "Data_1"
encodings, names = load_training_data(train_dir)
np.savez_compressed("buffalo.npz", encodings=encodings, names=names)

print(f"Saved {len(encodings)} face encodings to 'buffalo.npz'")








# using insightface buffalo_l , add new employee , remove old employee

# import os
# import cv2
# import hashlib
# import shutil
# import numpy as np
# from insightface.app import FaceAnalysis

# DATASET_DIR = "Data"
# NPZ_FILE = "face_data_s.npz"

# class EmployeeFaceManager:
#     def __init__(self, dataset_dir=DATASET_DIR, npz_file=NPZ_FILE):
#         self.dataset_dir = dataset_dir
#         self.npz_file = npz_file
#         self.app = FaceAnalysis(name='buffalo_s')
#         self.app.prepare(ctx_id=-1, det_size=(160, 160))  # CPU
#         self.encodings, self.names, self.hashes = self._load_data()

#     def _compute_hash(self, image_path):
#         with open(image_path, "rb") as f:
#             return hashlib.sha256(f.read()).hexdigest()

#     def _encode_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"[WARN] Cannot read: {image_path}")
#             return None
#         faces = self.app.get(img)
#         if faces:
#             return faces[0].embedding
#         print(f"[INFO] No face detected in: {image_path}")
#         return None

#     def _load_data(self):
#         if os.path.exists(self.npz_file):
#             data = np.load(self.npz_file, allow_pickle=True)
#             return list(data['encodings']), list(data['names']), list(data['hashes'])
#         return [], [], []

#     def _save_data(self):
#         np.savez_compressed(self.npz_file, encodings=self.encodings, names=self.names, hashes=self.hashes)
#         print(f"[SAVE] Data saved to {self.npz_file}")

#     def initialize_dataset(self):
#         self.encodings, self.names, self.hashes = [], [], []
#         for person in os.listdir(self.dataset_dir):
#             person_dir = os.path.join(self.dataset_dir, person)
#             if os.path.isdir(person_dir):
#                 for img in os.listdir(person_dir):
#                     img_path = os.path.join(person_dir, img)
#                     face_hash = self._compute_hash(img_path)
#                     embedding = self._encode_image(img_path)
#                     if embedding is not None:
#                         self.encodings.append(embedding)
#                         self.names.append(person)
#                         self.hashes.append(face_hash)
#         self._save_data()

#     def add_new_faces(self):
#         existing_hashes = set(self.hashes)
#         added = 0
#         for person in os.listdir(self.dataset_dir):
#             person_dir = os.path.join(self.dataset_dir, person)
#             if os.path.isdir(person_dir):
#                 for img in os.listdir(person_dir):
#                     img_path = os.path.join(person_dir, img)
#                     face_hash = self._compute_hash(img_path)
#                     if face_hash not in existing_hashes:
#                         embedding = self._encode_image(img_path)
#                         if embedding is not None:
#                             self.encodings.append(embedding)
#                             self.names.append(person)
#                             self.hashes.append(face_hash)
#                             added += 1
#         self._save_data() if added else print("[INFO] No new images found.")

#     def remove_employee(self, employee_name):
#     # Step 1: Remove employee folder from disk
#         employee_path = os.path.join(self.dataset_dir, employee_name)
#         if os.path.exists(employee_path) and os.path.isdir(employee_path):
#             shutil.rmtree(employee_path)
#             print(f"[FOLDER DELETE] Folder for '{employee_name}' deleted.")
#         else:
#             print(f"[INFO] Folder for '{employee_name}' not found.")

#         # Step 2: Remove their data from .npz
#         data = [(e, n, h) for e, n, h in zip(self.encodings, self.names, self.hashes) if n != employee_name]
#         if len(data) == len(self.encodings):
#             print(f"[INFO] No encodings found for: {employee_name}")
#             return
#         self.encodings, self.names, self.hashes = zip(*data) if data else ([], [], [])
#         self._save_data()
#         print(f"[REMOVE] Employee '{employee_name}' removed from database.")

# # ---------- MAIN CONTROL ----------
# def main():
#     manager = EmployeeFaceManager()

#     if not os.path.exists(NPZ_FILE):
#         print("[INIT] Creating new dataset...")
#         manager.initialize_dataset()
#     else:
#         while True:
#             print("\nChoose action:\n1. Add new employee/images\n2. Remove employee\n3. Exit")
#             choice = input("Enter choice (1/2/3): ").strip()
#             if choice == "1":
#                 manager.add_new_faces()
#             elif choice == "2":
#                 name = input("Enter employee name to remove: ").strip()
#                 manager.remove_employee(name)
#             elif choice == "3":
#                 break
#             else:
#                 print("[ERROR] Invalid option. Try again.")

# if __name__ == "__main__":
#     main()




# using MobileFaceNet ONNX model
# import os
# import cv2
# import numpy as np
# import onnxruntime as ort

# # Load MobileFaceNet ONNX model
# session = ort.InferenceSession("mobilefacenet.onnx", providers=['CPUExecutionProvider'])
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

# # Preprocessing function
# def preprocess(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Could not load image: {image_path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (112, 112))
#     img = img.astype(np.float32) / 255.0
#     img = (img - 0.5) / 0.5
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, axis=0)
#     return img

# # Extract embedding
# def get_embedding(image_path):
#     img = preprocess(image_path)
#     embedding = session.run([output_name], {input_name: img})[0]
#     return embedding[0]

# # Load dataset and extract embeddings
# def process_dataset(data_dir):
#     embeddings = []
#     labels = []

#     for person_name in os.listdir(data_dir):
#         person_dir = os.path.join(data_dir, person_name)
#         if not os.path.isdir(person_dir):
#             continue
#         for img_name in os.listdir(person_dir):
#             img_path = os.path.join(person_dir, img_name)
#             try:
#                 emb = get_embedding(img_path)
#                 embeddings.append(emb)
#                 labels.append(person_name)
#                 print(f"Processed: {img_path}")
#             except Exception as e:
#                 print(f"Skipping {img_path}: {e}")

#     embeddings = np.array(embeddings)
#     labels = np.array(labels)

#     # Save both in one .npz file
#     np.savez('face_data_embeddings.npz', embeddings=embeddings, labels=labels)
#     print(f"Saved {len(embeddings)} embeddings for {len(set(labels))} people.")

# # === Run it ===
# if __name__ == "__main__":
#     process_dataset("Data_1")  # Change "data" to your dataset folder path
