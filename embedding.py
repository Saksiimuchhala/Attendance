# import os
# import cv2
# import torch
# import joblib
# import numpy as np
# from PIL import Image
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
# from torchvision import transforms

# # Device and model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Image transform (same for all embedding generation)
# transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

# def extract_face_embedding(image_bgr):
#     """
#     Extracts a face embedding from a BGR image.
#     Returns embedding as numpy array, or None if no face detected.
#     """
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces) == 0:
#         return None

#     x, y, w, h = faces[0]  # Use the first detected face
#     face_img = image_bgr[y:y + h, x:x + w]
#     face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#     face_tensor = transform(face_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         embedding = model(face_tensor)

#     return embedding.squeeze().cpu().numpy()

# def create_embeddings_from_folder(data_dir='static/faces'):
#     """
#     Loops through each person folder in data_dir, processes images,
#     and stores average embedding per person.
#     """
#     embedding_db = {}

#     for person_name in os.listdir(data_dir):
#         person_path = os.path.join(data_dir, person_name)
#         if not os.path.isdir(person_path):
#             continue

#         embeddings = []

#         for img_name in os.listdir(person_path):
#             img_path = os.path.join(person_path, img_name)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue

#             embedding = extract_face_embedding(img)
#             if embedding is not None:
#                 embeddings.append(embedding)

#         if embeddings:
#             embedding_db[person_name] = np.mean(embeddings, axis=0)

#     return embedding_db

# def save_embeddings(embedding_db, save_path='face_embeddings.pkl'):
#     """
#     Saves the embedding database to a .pkl file.
#     """
#     joblib.dump(embedding_db, save_path)
#     print(f"Saved embeddings for {len(embedding_db)} employees to {save_path}.")

# def main():
#     data_dir = 'Data'
#     embedding_db = create_embeddings_from_folder(data_dir)
#     save_embeddings(embedding_db)

# if __name__ == '__main__':
#     main()


import os
import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

# Device and model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Image transform for embedding extraction
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

def extract_embedding_from_cropped_face(image_bgr):
    """
    Extract embedding from a cropped face image (BGR).
    """
    face_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding.squeeze().cpu().numpy()

def create_embeddings_from_folder(data_dir='Data'):
    """
    Create embeddings from cropped face images in folder structure:
    Data/
      employee_1/
        img1.jpg
        img2.jpg
      employee_2/
        img1.jpg
        ...
    Returns dict with employee_name -> list of embeddings
    """
    embedding_db = {}

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            embedding = extract_embedding_from_cropped_face(img)
            embeddings.append(embedding)

        if embeddings:
            embedding_db[person_name] = embeddings
            print(f"Processed {len(embeddings)} images for {person_name}")

    return embedding_db

def save_embeddings(embedding_db, save_path='face_embeddings.pkl'):
    """
    Save the embedding database to disk with joblib.
    """
    joblib.dump(embedding_db, save_path)
    print(f"Saved embeddings for {len(embedding_db)} employees to {save_path}.")

def main():
    data_dir = 'Data'  # Your dataset root folder
    embedding_db = create_embeddings_from_folder(data_dir)
    save_embeddings(embedding_db)

if __name__ == '__main__':
    main()
