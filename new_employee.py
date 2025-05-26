# Script to capture live face images for a new employee using MTCNN
import cv2
import os
import time
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Initialize MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # Detect only 1 face per frame

def validate_name(name):
    return name and name[0].isupper()

def create_employee_folder(base_path, employee_name):
    path = os.path.join(base_path, employee_name)
    os.makedirs(path, exist_ok=True)
    return path

def capture_faces_live(employee_name, output_folder="Data", duration=18, max_images=36, fps=2):
    if not validate_name(employee_name):
        print("Error: The first letter of the employee's name must be capitalized.")
        return

    employee_path = create_employee_folder(output_folder, employee_name)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print(f"Capturing face images for {employee_name}...")

    count = 0
    interval = 1 / fps
    start_time = time.time()

    while (time.time() - start_time < duration) and (count < max_images):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert to RGB for MTCNN
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]  # frame is still BGR

                if face.size == 0:
                    continue  # skip empty crops

                existing_images = [f for f in os.listdir(employee_path) if f.endswith('.jpg')]
                existing_numbers = [
                int(f.split('_')[-1].split('.')[0]) for f in existing_images if f.startswith(employee_name) and f.split('_')[-1].split('.')[0].isdigit()]
                start_index = max(existing_numbers, default=0) + 1
                img_name = os.path.join(employee_path, f"{employee_name}_{start_index + count}.jpg")
                
                cv2.imwrite(img_name, face)
                print(f"✅ Saved: {img_name}")
                count += 1

                if count >= max_images:
                    break

        # Show camera feed (optional)
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(interval)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {count} face images for {employee_name}.")

if __name__ == "__main__":
    name = input("Enter employee name (first letter must be capital): ").strip()
    capture_faces_live(name, output_folder="Data", duration=10, max_images=20, fps=2)
