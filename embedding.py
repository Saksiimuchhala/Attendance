# import os
# from PIL import Image
# import torch
# from torchvision import transforms
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
# import joblib

# # Setup
# data_dir = "Data"
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

# known_embeddings = []
# known_names = []

# for person in os.listdir(data_dir):
#     person_dir = os.path.join(data_dir, person)
#     if not os.path.isdir(person_dir): continue

#     for img_name in os.listdir(person_dir):
#         img_path = os.path.join(person_dir, img_name)
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0).to(device)
#         embedding = resnet(img_tensor).detach().cpu().numpy()[0]

#         known_embeddings.append(embedding)
#         known_names.append(person)

# # Save embeddings
# joblib.dump((known_embeddings, known_names), "face_embeddings.pkl")
# print("Embeddings saved.")


# create_embeddings.py
import os
import face_recognition
import joblib
from PIL import Image
import numpy as np

def create_face_encodings(data_dir, target_size=(400, 400)):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            try:
                # Open and resize image using PIL
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Ensure RGB format
                    img = img.resize(target_size)
                    image_np = np.array(img)

                # Get face encodings
                encodings = face_recognition.face_encodings(image_np)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    return known_encodings, known_names

if __name__ == "__main__":
    data_dir = "Data"
    encodings, names = create_face_encodings(data_dir)

    # Save encodings and names using joblib
    joblib.dump((encodings, names), "saved_encodings.pkl")
    print("Encodings saved to saved_encodings.pkl")

