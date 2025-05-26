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
# import os
# import face_recognition
# import joblib
# from PIL import Image
# import numpy as np

# def create_face_encodings(data_dir, target_size=(400, 400)):
#     known_encodings = []
#     known_names = []

#     for person_name in os.listdir(data_dir):
#         person_folder = os.path.join(data_dir, person_name)
#         if not os.path.isdir(person_folder):
#             continue

#         for img_name in os.listdir(person_folder):
#             img_path = os.path.join(person_folder, img_name)

#             try:
#                 # Open and resize image using PIL
#                 with Image.open(img_path) as img:
#                     img = img.convert("RGB")  # Ensure RGB format
#                     img = img.resize(target_size)
#                     image_np = np.array(img)

#                 # Get face encodings
#                 encodings = face_recognition.face_encodings(image_np)

#                 if encodings:
#                     known_encodings.append(encodings[0])
#                     known_names.append(person_name)

#             except Exception as e:
#                 print(f"Failed to process {img_path}: {e}")

#     return known_encodings, known_names

# if __name__ == "__main__":
#     data_dir = "Data"
#     encodings, names = create_face_encodings(data_dir)

#     # Save encodings and names using joblib
#     joblib.dump((encodings, names), "saved_encodings.pkl")
#     print("Encodings saved to saved_encodings.pkl")

import os
import face_recognition
import joblib
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from collections import defaultdict

def augment_face_image(image_np):
    """
    Create augmented versions of face images to improve generalization
    """
    augmented_images = [image_np]  # Original image
    
    # Convert to PIL for easier manipulation
    pil_image = Image.fromarray(image_np)
    
    try:
        # 1. Slight brightness variations
        brightness_enhancer = ImageEnhance.Brightness(pil_image)
        bright_image = brightness_enhancer.enhance(1.2)  # 20% brighter
        dim_image = brightness_enhancer.enhance(0.8)     # 20% dimmer
        augmented_images.extend([np.array(bright_image), np.array(dim_image)])
        
        # 2. Slight contrast variations
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        high_contrast = contrast_enhancer.enhance(1.3)   # Higher contrast
        low_contrast = contrast_enhancer.enhance(0.7)    # Lower contrast
        augmented_images.extend([np.array(high_contrast), np.array(low_contrast)])
        
        # 3. Slight rotations (small angles)
        for angle in [-3, 3]:  # Small rotations
            rotated = pil_image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
            augmented_images.append(np.array(rotated))
        
        # 4. Horizontal flip (if face is not too asymmetric)
        flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(np.array(flipped))
        
    except Exception as e:
        print(f"Warning: Augmentation failed: {e}")
    
    return augmented_images

def create_face_encodings_robust(data_dir, target_size=(200, 200), use_augmentation=True, max_images_per_person=15):
    known_encodings = []
    known_names = []
    person_encodings = defaultdict(list)
    
    print("Creating robust face encodings...")
    
    for person_name in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing {person_name}...")
        person_encoding_count = 0
        
        image_files = [f for f in os.listdir(person_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_name in image_files:
            if person_encoding_count >= max_images_per_person:
                break
                
            img_path = os.path.join(person_folder, img_name)

            try:
                # Load and preprocess image
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    image_np = np.array(img)

                # Create base encoding with high jitter for robustness
                base_encodings = face_recognition.face_encodings(image_np, num_jitters=20)
                
                if not base_encodings:
                    print(f"No face found in {img_path}")
                    continue
                
                base_encoding = base_encodings[0]
                known_encodings.append(base_encoding)
                known_names.append(person_name)
                person_encodings[person_name].append(base_encoding)
                person_encoding_count += 1
                
                # Create augmented versions if enabled and we haven't hit the limit
                if use_augmentation and person_encoding_count < max_images_per_person:
                    augmented_images = augment_face_image(image_np)
                    
                    for aug_idx, aug_image in enumerate(augmented_images[1:]):  # Skip original
                        if person_encoding_count >= max_images_per_person:
                            break
                            
                        aug_encodings = face_recognition.face_encodings(aug_image, num_jitters=5)
                        
                        if aug_encodings:
                            known_encodings.append(aug_encodings[0])
                            known_names.append(person_name)
                            person_encodings[person_name].append(aug_encodings[0])
                            person_encoding_count += 1

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
        
        print(f"  Created {person_encoding_count} encodings for {person_name}")

    # Calculate optimal tolerance
    tolerance_info = calculate_optimal_tolerance(person_encodings)
    
    return known_encodings, known_names, person_encodings, tolerance_info

def calculate_optimal_tolerance(person_encodings):
    """
    Calculate optimal tolerance based on dataset characteristics
    """
    intra_distances = []  # Same person distances
    inter_distances = []  # Different person distances
    
    persons = list(person_encodings.keys())
    
    # Calculate intra-person distances
    for person, encodings in person_encodings.items():
        if len(encodings) > 1:
            for i in range(len(encodings)):
                for j in range(i+1, len(encodings)):
                    dist = face_recognition.face_distance([encodings[i]], encodings[j])[0]
                    intra_distances.append(dist)
    
    # Calculate inter-person distances (sample to avoid too many comparisons)
    sample_size = min(10, len(persons))
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            person1_encodings = person_encodings[persons[i]]
            person2_encodings = person_encodings[persons[j]]
            
            if person1_encodings and person2_encodings:
                # Compare first encoding of each person
                dist = face_recognition.face_distance([person1_encodings[0]], person2_encodings[0])[0]
                inter_distances.append(dist)
    
    if intra_distances and inter_distances:
        avg_intra = np.mean(intra_distances)
        max_intra = np.max(intra_distances)
        avg_inter = np.mean(inter_distances)
        min_inter = np.min(inter_distances)
        
        # Calculate tolerance with safety margin
        # Should be higher than max intra-distance but lower than min inter-distance
        safety_margin = 0.05
        suggested_tolerance = min(0.8, max(0.45, max_intra + safety_margin))
        
        # If there's overlap between intra and inter distances, use a compromise
        if max_intra >= min_inter:
            suggested_tolerance = (avg_intra + avg_inter) / 2
            print("Warning: Some overlap between same-person and different-person distances detected")
        
        return {
            'suggested_tolerance': suggested_tolerance,
            'avg_intra': avg_intra,
            'max_intra': max_intra,
            'avg_inter': avg_inter,
            'min_inter': min_inter,
            'separation_quality': min_inter - max_intra
        }
    
    return {'suggested_tolerance': 0.6}

if __name__ == "__main__":
    data_dir = "Data"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        exit(1)
    
    # Create robust encodings with augmentation
    encodings, names, person_encodings, tolerance_info = create_face_encodings_robust(
        data_dir, 
        target_size=(200, 200),  # Smaller size for consistency
        use_augmentation=True,
        max_images_per_person=12
    )
    
    if not encodings:
        print("No face encodings were created. Please check your dataset.")
        exit(1)
    
    # Save both formats
    joblib.dump((encodings, names), "saved_encodings.pkl")
    
    enhanced_data = {
        'encodings': encodings,
        'names': names,
        'person_encodings': dict(person_encodings),
        'tolerance_info': tolerance_info
    }
    joblib.dump(enhanced_data, "enhanced_encodings.pkl")
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("ENCODING ANALYSIS")
    print("="*60)
    print(f"Total people: {len(person_encodings)}")
    print(f"Total encodings: {len(encodings)}")
    print(f"Average encodings per person: {len(encodings)/len(person_encodings):.1f}")
    
    if 'suggested_tolerance' in tolerance_info:
        print(f"\nSuggested tolerance: {tolerance_info['suggested_tolerance']:.3f}")
        
        if 'avg_intra' in tolerance_info:
            print(f"Average same-person distance: {tolerance_info['avg_intra']:.3f}")
            print(f"Maximum same-person distance: {tolerance_info['max_intra']:.3f}")
            print(f"Average different-person distance: {tolerance_info['avg_inter']:.3f}")
            print(f"Minimum different-person distance: {tolerance_info['min_inter']:.3f}")
            print(f"Separation quality: {tolerance_info['separation_quality']:.3f}")
            
            if tolerance_info['separation_quality'] > 0.1:
                print("✅ Good separation between people - should work well")
            elif tolerance_info['separation_quality'] > 0.05:
                print("⚠️  Moderate separation - may need fine-tuning")
            else:
                print("❌ Poor separation - consider adding more diverse images")
    
    print(f"\nPer-person encoding count:")
    for person, encodings_list in person_encodings.items():
        print(f"  {person}: {len(encodings_list)} encodings")
    
    print(f"\nFiles saved:")
    print(f"  - saved_encodings.pkl (compatible format)")
    print(f"  - enhanced_encodings.pkl (with metadata)")
    print("\n✅ Robust encoding creation completed!")