# using mtcnn for face detection

import cv2
import os
import time
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Initialize MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # Capture only 1 face per frame

def validate_name(name):
    return name and name[0].isupper()

def create_employee_folder(base_path, employee_name):
    path = os.path.join(base_path, employee_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_next_index(folder_path):
    existing = [int(f.split('.')[0]) for f in os.listdir(folder_path)
                if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    return max(existing, default=0) + 1

def capture_faces_fixed_count(employee_name, output_folder="Data", total_images=36, wait_between=0.5):
    if not validate_name(employee_name):
        print("Error: Employee name must start with a capital letter.")
        return

    employee_path = create_employee_folder(output_folder, employee_name)
    next_index = get_next_index(employee_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print(f"📸 Starting capture for {employee_name} ({total_images} images)...")

    count = 0
    while count < total_images:
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
                face = frame[y1:y2, x1:x2]  # BGR

                if face.size == 0:
                    continue

                img_name = os.path.join(employee_path, f"{next_index + count}.jpg")
                cv2.imwrite(img_name, face)
                print(f"✅ Saved: {img_name}")
                count += 1
                break  # Only save one face per frame

        # Show webcam feed (optional)
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(wait_between)  # Wait between captures to reduce duplicates

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Captured {count} face images for {employee_name}.")

if __name__ == "__main__":
    name = input("Enter employee name (first letter capitalized): ").strip()
    capture_faces_fixed_count(name, output_folder="Data", total_images=30, wait_between=0.3)



# using face_location for getting images 
# import cv2
# import os
# import time
# import numpy as np
# import face_recognition

# def validate_name(name):
#     return name and name[0].isupper()

# def create_employee_folder(base_path, employee_name):
#     path = os.path.join(base_path, employee_name)
#     os.makedirs(path, exist_ok=True)
#     return path

# def get_next_index(folder_path):
#     existing = [int(f.split('.')[0]) for f in os.listdir(folder_path)
#                 if f.endswith('.jpg') and f.split('.')[0].isdigit()]
#     return max(existing, default=0) + 1

# def capture_faces_fixed_count(employee_name, output_folder="Data", total_images=36, wait_between=0.5):
#     if not validate_name(employee_name):
#         print("Error: Employee name must start with a capital letter.")
#         return

#     employee_path = create_employee_folder(output_folder, employee_name)
#     next_index = get_next_index(employee_path)

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Cannot access webcam.")
#         return

#     print(f"📸 Starting capture for {employee_name} ({total_images} images)...")

#     count = 0
#     while count < total_images:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame.")
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)

#         if face_locations:
#             for (top, right, bottom, left) in face_locations:
#                 face = frame[top:bottom, left:right]

#                 if face.size == 0:
#                     continue

#                 img_name = os.path.join(employee_path, f"{next_index + count}.jpg")
#                 cv2.imwrite(img_name, face)
#                 print(f"✅ Saved: {img_name}")
#                 count += 1
#                 break  # Save only one face per frame

#         # Show webcam feed
#         cv2.imshow("Capturing Faces (press 'q' to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         time.sleep(wait_between)

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"✅ Done! Captured {count} face images for {employee_name}.")

# if __name__ == "__main__":
#     name = input("Enter employee name (first letter capitalized): ").strip()
#     capture_faces_fixed_count(name, output_folder="Data", total_images=20, wait_between=0.3)
