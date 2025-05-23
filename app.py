import cv2
import os
import base64
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load face detection model
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Today's date
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Extract faces from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return faces

# Identify employee using ML model
def identify_face(face_array):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(face_array)
    except FileNotFoundError:
        return None  # Model missing, return None


# Train model
def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')



# Extract attendance data
def extract_attendance():
    df = pd.read_csv(csv_file)
    return df['Name'], df['ID'], df['Check-In Time'], df['Check-Out Time'], len(df)


# Add this as a global variable at the start of your script
last_attendance = {}  # Dictionary to store last attendance time for each person
employee_status = {}  # Dictionary to track employee check-in/out status

# Add Check-In / Check-Out
def update_attendance(name):
    try:
        global last_attendance, employee_status
        current_time = datetime.now()
        
        # Check cooldown only for check-in
        if name in last_attendance and name in employee_status:
            if employee_status[name] == 'Checked In':
                time_diff = current_time - last_attendance[name]
                if time_diff.total_seconds() < 30:  # 30 second cooldown
                    return "Attendance already marked recently"
        
        # Create Attendance directory if it doesn't exist
        os.makedirs('Attendance', exist_ok=True)
        
        # Generate filename with current date only
        csv_file = f'Attendance/Attendance-{current_time.strftime("%Y-%m-%d")}.csv'
        
        # Create DataFrame with attendance data
        data = {
            'Name': [name],
            'Time': [current_time.strftime("%H:%M:%S")],
            'Date': [current_time.strftime("%Y-%m-%d")]
        }
        new_df = pd.DataFrame(data)
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        
        df.to_csv(csv_file, index=False)
        last_attendance[name] = current_time
        
        # Update status
        status = 'Checked Out' if name in employee_status and employee_status[name] == 'Checked In' else 'Checked In'
        employee_status[name] = status
        
        return f"{name} successfully recorded: {status}"
        
    except Exception as e:
        print(f"Error updating attendance: {str(e)}")
        return "Error updating attendance"
    

# Retain original routes
@app.route('/')
def home():
    return  "Hello World!"

@app.route('/add', methods=['POST'])
def add():
    # Parse JSON raw data
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    # Safely get 'newuserid' from the JSON data
    newuserid = data.get('userId')
    if not newuserid:
        return jsonify({"error": "Missing 'userId' in the request"}), 400

    # Safely get 'images' from the JSON data
    images = data.get('images')
    if not images or not isinstance(images, list):
        return jsonify({"error": "Missing or invalid 'images' in the request"}), 400

    userimagefolder = f'static/faces/{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)

    for idx, jsonstringbase64Image in enumerate(images):
        # Remove the 'data:image/png;base64,' prefix if present
        if jsonstringbase64Image.startswith('data:image'):
            jsonstringbase64Image = jsonstringbase64Image.split(',')[1]
        
        # Ensure the Base64 string is properly padded
        missing_padding = len(jsonstringbase64Image) % 4
        if missing_padding:
            jsonstringbase64Image += '=' * (4 - missing_padding)
        
        try:
            # Decode the Base64 string into an image
            image_data = base64.b64decode(jsonstringbase64Image)
            np_array = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Check if the image was successfully decoded
            if frame is None:
                return jsonify({"error": f"Failed to decode image at index {idx}. The image data might be invalid."}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to decode image at index {idx}: {str(e)}"}), 400

        faces = extract_faces(frame)
        if len(faces) == 0:
            return jsonify({"error": f"No faces detected in the image at index {idx}."}), 400

        (x, y, w, h) = faces[0]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 40), (255, 0, 0), -1)
        
        # Display "Registering" below the bounding box
        cv2.putText(frame, "Registering", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save image inside bounding box
        face_crop = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (100, 100))
        cv2.imwrite(f'{userimagefolder}/{newuserid}_{idx}.jpg', face_resized)

    train_model()
    return jsonify({"message": f"Successfully processed {len(images)} images for user {newuserid}."}), 200

@app.route('/check_in_out', methods=['POST'])
def start():
    # Parse JSON raw data
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400
    
    jsonstringbase64Image = data.get('image')

    if not jsonstringbase64Image:
        return jsonify({"error": "Missing 'image' in the request"}), 400
    
    if jsonstringbase64Image.startswith('data:image'):
        jsonstringbase64Image = jsonstringbase64Image.split(',')[1]

    missing_padding = len(jsonstringbase64Image) % 4
    if missing_padding:
        jsonstringbase64Image += '=' * (4 - missing_padding)

    try:
        # Decode the Base64 string into an image
        image_data = base64.b64decode(jsonstringbase64Image)
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Check if the image was successfully decoded
        if frame is None:
            return jsonify({"error": f"Failed to decode image. The image data might be invalid."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    faces = extract_faces(frame)
    if len(faces) == 0:
        return jsonify({"error": f"No faces detected in the image."}), 400   
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Identify the face
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
        
            # Convert identified_person to a JSON-serializable format
            if isinstance(identified_person, np.ndarray):
                identified_person = identified_person.tolist()  # Convert NumPy array to list
            elif isinstance(identified_person, (int, float)):
                identified_person = str(identified_person)  # Convert to string if it's a number
    
        return jsonify({"message": f"Successfully identified {identified_person}.", "userId": identified_person}), 200
    else:
        return jsonify({"error": "No faces detected in the image."}), 400
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
