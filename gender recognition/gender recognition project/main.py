import cv2
import numpy as np
import os
import requests
import pickle

# --- Configuration ---
MODEL_DIR = "models"
DATA_DIR = "data"
LEARNED_DATA_FILE = os.path.join(DATA_DIR, "learned_data.pkl")

# Files for the pre-trained gender model
PROTO_FILE = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
MODEL_FILE = os.path.join(MODEL_DIR, "gender_net.caffemodel")
PROTO_URL = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt"
MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"

# Files for Face Detection
# We usually use the one included in cv2.data.haarcascades

# --- Helper Functions ---

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, filepath):
    if os.path.exists(filepath):
         if os.path.getsize(filepath) < 1000:
            print(f"File {filepath} looks corrupted. Re-downloading...")
            os.remove(filepath)
         else:
            return True
            
    print(f"Downloading {os.path.basename(filepath)}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

# --- Classes ---

class OnlineLearner:
    """
    Learns from user feedback using HOG features + KNN.
    """
    def __init__(self, k=3):
        self.k = k
        self.hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
        self.knn = cv2.ml.KNearest_create()
        self.samples = []
        self.labels = []
        self.trained = False
        self.load_data()

    def load_data(self):
        if os.path.exists(LEARNED_DATA_FILE):
            try:
                with open(LEARNED_DATA_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.samples = data['samples']
                    self.labels = data['labels']
                    if len(self.samples) > 0:
                        self.train()
                print(f"Loaded {len(self.samples)} learned samples.")
            except Exception as e:
                print(f"Could not load data: {e}")

    def save_data(self):
        ensure_directory(DATA_DIR)
        with open(LEARNED_DATA_FILE, 'wb') as f:
            pickle.dump({'samples': self.samples, 'labels': self.labels}, f)
        print("Learned data saved.")

    def compute_features(self, face_img):
        # Resize to fixed size for HOG
        resized = cv2.resize(face_img, (64, 128))
        # Compute descriptors
        hist = self.hog.compute(resized)
        return hist.flatten()

    def add_sample(self, face_img, label_int):
        """
        label_int: 0 for Male, 1 for Female
        """
        features = self.compute_features(face_img)
        self.samples.append(features)
        self.labels.append(label_int)
        self.train() # Retrain immediately for "instant" feel
        self.save_data()

    def train(self):
        if len(self.samples) < self.k:
            return # Not enough data yet
        
        train_data = np.array(self.samples, dtype=np.float32)
        train_labels = np.array(self.labels, dtype=np.int32)
        
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        self.trained = True

    def predict(self, face_img):
        if not self.trained:
            return None, 0.0
            
        features = self.compute_features(face_img)
        features = np.array([features], dtype=np.float32)
        
        ret, results, neighbours, dist = self.knn.findNearest(features, self.k)
        
        # Determine confidence based on distance or unanimity
        # For simplicity, if majority agrees, we accept it.
        prediction = int(results[0][0])
        
        # Simple distance check - if very far, might be unknown, but we want to override base model
        # so we trust it if it's reasonably close.
        return prediction, 1.0 # Returning simplified confidence

# --- Main Application ---

def main():
    ensure_directory(MODEL_DIR)
    
    # 1. Download Models
    try:
        download_file(PROTO_URL, PROTO_FILE)
        download_file(MODEL_URL, MODEL_FILE)
    except Exception as e:
        print("Failed to setup models. Check internet connection.")
        return

    # 2. Setup Detectors
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    print("Loading Deep Learning Model...")
    gender_net = cv2.dnn.readNet(MODEL_FILE, PROTO_FILE)
    
    gender_list = ['Male', 'Female']
    
    # 3. Setup Learner
    learner = OnlineLearner()

    # 4. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("="*50)
    print(" GENDER RECOGNITION APP STARTED")
    print("="*50)
    print(" [ Controls ]")
    print("  'M' : Correct selection as MALE (Learn)")
    print("  'F' : Correct selection as FEMALE (Learn)")
    print("  'C' : Clear all learned data")
    print("  'Q' : Quit")
    print("="*50)

    last_face_img = None # Store last face for learning

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # We will process the largest face primarily for the 'learning' interaction to be intuitive
        target_face = None
        target_rect = None
        max_area = 0

        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                target_rect = (x, y, w, h)
                
            # --- Inference for all faces ---
            face_img = frame[y:y+h, x:x+w].copy()
            
            # 1. Ask Learner first
            learned_label, confidence = learner.predict(face_img)
            
            source = "AI"
            if learned_label is not None:
                # Use learned result
                gender = gender_list[learned_label]
                source = "Learned"
                color = (0, 255, 0) # Green for learned
            else:
                # 2. Fallback to Standard Model
                # Blob for Caffe model
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                           (78.4263377603, 87.7689143744, 114.895847746), 
                                           swapRB=False)
                gender_net.setInput(blob)
                preds = gender_net.forward()
                i = preds[0].argmax()
                gender = gender_list[i]
                prob = preds[0][i]
                color = (255, 0, 0) # Blue for default AI

            label = f"{gender} ({source})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Store the target face (largest) for user feedback
        if target_rect:
            x, y, w, h = target_rect
            last_face_img = frame[y:y+h, x:x+w].copy()

        cv2.imshow('Gender Recognition & Learning', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            if last_face_img is not None:
                print("Learning: This face is MALE")
                learner.add_sample(last_face_img, 0) # 0 for Male
                # Visual feedback
                cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0, 255, 0), -1)
                cv2.putText(frame, "LEARNED: MALE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Gender Recognition & Learning', frame)
                cv2.waitKey(500)
            else:
                print("No face detected to learn!")
        elif key == ord('f'):
            if last_face_img is not None:
                print("Learning: This face is FEMALE")
                learner.add_sample(last_face_img, 1) # 1 for Female
                # Visual feedback
                cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0, 255, 0), -1)
                cv2.putText(frame, "LEARNED: FEMALE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Gender Recognition & Learning', frame)
                cv2.waitKey(500)
            else:
                print("No face detected to learn!")
        elif key == ord('c'):
            learner.samples = []
            learner.labels = []
            learner.trained = False
            learner.save_data()
            print("Memory cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
