import os
import cv2
import joblib
import numpy as np
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load stored embeddings
def load_embeddings(path='face_embeddings.pkl'):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print("Embedding file not found!")
        return {}

# Get embedding from image crop
def get_embedding_from_image(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().cpu().numpy()

def detect_and_save_all_faces(image_path, save_dir="detect"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No faces detected.")
        return []

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_crop = img[y:y+h, x:x+w]
        save_path = os.path.join(save_dir, f"{i}.jpg")
        cv2.imwrite(save_path, face_crop)
        saved_paths.append(save_path)
        print(f"✅ Saved face {i} to: {save_path}")

    return saved_paths

# Recognize face in a single image
def recognize_faces_in_image(image_path, embeddings_db, threshold=0.6):
    # Step 1: Detect and save all face crops
    cropped_face_paths = detect_and_save_all_faces(image_path, save_dir="detect")
    if not cropped_face_paths:
        print("⚠️ No faces to recognize.")
        return

    embeddings = joblib.load('face_embeddings.pkl')
    print("Stored employees:", list(embeddings.keys()))


    # Step 2: Loop through each cropped face
    for i, crop_path in enumerate(cropped_face_paths, start=1):
        img = cv2.imread(crop_path)
        if img is None:
            print(f"❌ Failed to load cropped face: {crop_path}")
            continue

        resized = cv2.resize(img, (160, 160))
        embedding = get_embedding_from_image(resized)
        if embedding is None:
            print(f"⚠️ Could not extract embedding for: {crop_path}")
            continue

        # Step 3: Compare to embeddings DB
        best_match = "Unknown"
        best_score = 0

        for name, emb_list in embeddings_db.items():
            sims = [cosine_similarity([embedding], [e])[0][0] for e in emb_list]
            max_sim = max(sims)
            print(f"Checking {name}: Max similarity = {max_sim:.3f}")
            if max_sim > best_score:
                best_score = max_sim
                best_match = name

            if best_score < threshold:
                best_match = "Unknown"

        print(f"Face {i}: {best_match} (Similarity: {best_score:.3f})")

    print("✅ Face recognition complete.")


if __name__ == "__main__":
    # Hardcoded image path here:
    image_path = r"D:\Sakshi muchhala\attendance\group.jpeg"

    embeddings_db = load_embeddings('face_embeddings.pkl')
    recognize_faces_in_image(image_path, embeddings_db)
