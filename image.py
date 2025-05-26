
# This script detects multiple faces in an image or from a live camera feed using MTCNN.
import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import time

# Hardcoded configuration
mode = "live"  # Change to "image" or "live"
path = r"D:\Sakshi muchhala\attendance\group3.jpeg"  # Used only in 'image' mode

# Initialize MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Ensure output folder exists
os.makedirs("detect", exist_ok=True)

def save_faces_from_image(image_path):
    img = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        print("⚠️ No faces detected.")
        return

    img_np = np.array(img)
    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(b) for b in box]
        face = img_np[y1:y2, x1:x2]
        save_path = os.path.join("detect", f"face_{i}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved face {i} to {save_path}")

def save_faces_from_camera():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    face_count = 0
    start_time = time.time() 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]
                face_count += 1
                save_path = os.path.join("detect", f"cam_face_{frame_count}_{face_count}.jpg")
                cv2.imwrite(save_path, face)

        # Optional: display live detection
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Live Face Detection (Press 'q' to quit)", frame)
        if time.time() - start_time > 3:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Extracted {face_count} faces from live camera.")

# Run the selected mode
if mode == 'image':
    save_faces_from_image(path)
elif mode == 'live':
    save_faces_from_camera()
else:
    print("❌ Invalid mode. Use 'image' or 'live'.")
