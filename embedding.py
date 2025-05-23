# import os
# import cv2
# import joblib
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

# def get_embedding(img_path):
#     img_bgr = cv2.imread(img_path)
#     if img_bgr is None:
#         return None

#     # Since images are cropped faces, no face detection needed
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_rgb)
#     img_tensor = transform(img_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         embedding = model(img_tensor).squeeze().cpu().numpy()
#     return embedding

# DATA_DIR = 'Data'  # Path to your dataset folder with subfolders per employee
# embedding_dict = {}

# for person in os.listdir(DATA_DIR):
#     person_path = os.path.join(DATA_DIR, person)
#     if not os.path.isdir(person_path):
#         continue

#     embeddings = []
#     for img_name in os.listdir(person_path):
#         img_path = os.path.join(person_path, img_name)
#         emb = get_embedding(img_path)
#         if emb is not None:
#             embeddings.append(emb)

#     if embeddings:
#         embedding_dict[person] = embeddings
#         print(f"{person}: {len(embeddings)} embeddings saved.")
#     else:
#         print(f"{person}: No valid images found.")

# print(embedding_dict)
# joblib.dump(embedding_dict, 'face_embeddings.pkl')
# print("Embeddings saved.")

# create_embeddings.py
import os
import face_recognition
import joblib
from PIL import Image
import numpy as np

def create_face_encodings(data_dir):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

    return known_encodings, known_names

if __name__ == "__main__":
    data_dir = "Data"
    encodings, names = create_face_encodings(data_dir)
    
    # Save encodings and names using joblib
    joblib.dump((encodings, names), "saved_encodings.pkl")
    print("Encodings saved to saved_encodings.pkl")
