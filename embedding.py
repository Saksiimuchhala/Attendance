# create encodings using face_recognition and save in joblib format
# import os
# import face_recognition
# from joblib import dump
# import numpy as np
# import cv2

# def load_training_data(train_dir):
#     face_encodings = []
#     face_names = []
    
#     for person_name in os.listdir(train_dir):
#         person_dir = os.path.join(train_dir, person_name)
#         if os.path.isdir(person_dir):
#             for img_name in os.listdir(person_dir):
#                 img_path = os.path.join(person_dir, img_name)
#                 image = face_recognition.load_image_file(img_path)
#                 # image = cv2.resize(image, (100, 100))
#                 face_locations = face_recognition.face_locations(image)
#                 encodings = face_recognition.face_encodings(image, face_locations)
#                 if len(encodings) > 0:
#                     face_encodings.append(encodings[0])
#                     face_names.append(person_name)
#                 else:
#                     print(f"No face encoding found in {img_path}")

#     return face_encodings, face_names

# # Load and save
# train_dir = "Data"
# encodings, names = load_training_data(train_dir)
# dump((encodings, names), "face_data.joblib")

# print(f"Saved {len(encodings)} face encodings to 'face_data.joblib'")


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
np.savez_compressed("face_data.npz", encodings=encodings, names=names)

print(f"Saved {len(encodings)} face encodings to 'face_data_1.npz'")

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
