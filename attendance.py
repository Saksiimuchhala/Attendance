import os
import cv2
import time
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

EMBEDDING_DB = 'face_embeddings.pkl'
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Load embeddings
def load_embeddings():
    if os.path.exists(EMBEDDING_DB):
        return joblib.load(EMBEDDING_DB)
    return {}

# FaceNet Model & Preprocessing
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

def get_embedding_from_image(image_bgr):
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# Attendance update logic
def update_attendance(name):
    current_time = datetime.now()
    csv_file = f'Attendance/Attendance-{current_time.strftime("%Y-%m-%d")}.csv'
    data = {'Name': [name], 'Time': [current_time.strftime("%H:%M:%S")], 'Date': [current_time.strftime("%Y-%m-%d")]}
    new_df = pd.DataFrame(data)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(csv_file, index=False)

# Main video capture and recognition
def recognize_from_camera(duration=7):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    # Use explicit FOURCC for compatibility (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    known_embeddings = load_embeddings()
    end_time = time.time() + duration

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_crop, (160, 160))
                embedding = get_embedding_from_image(face_resized)

                if embedding is None:
                    continue

                best_match = "Unknown"
                highest_similarity = 0.3  # threshold can be tuned

                # Compare with all embeddings per person
                for name, emb_list in known_embeddings.items():
                    emb_array = np.array(emb_list)  # (num_images, embedding_dim)
                    sims = cosine_similarity([embedding], emb_array)  # (1, num_images)
                    max_sim = sims.max()

                    if max_sim > highest_similarity:
                        highest_similarity = max_sim
                        best_match = name

                # Mark attendance and draw box
                if best_match != "Unknown":
                    update_attendance(best_match)
                    color = (0, 255, 0)  # green
                else:
                    color = (0, 0, 255)  # red

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"Error processing face: {e}")

        out.write(frame)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")

if __name__ == '__main__':
    recognize_from_camera(duration=7)
