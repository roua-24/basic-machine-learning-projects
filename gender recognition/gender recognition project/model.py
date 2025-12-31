import cv2
import numpy as np
import os
import requests
import pickle
import threading
import time

# --- Configuration ---
MODEL_DIR = "models"
DATA_DIR = "data"
LEARNED_DATA_FILE = os.path.join(DATA_DIR, "learned_data.pkl")

PROTO_FILE = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
MODEL_FILE = os.path.join(MODEL_DIR, "gender_net.caffemodel")

PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
MODEL_URL = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, filepath):
    if os.path.exists(filepath):
         if os.path.getsize(filepath) < 1000:
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
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

class OnlineLearner:
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
            except Exception:
                pass

    def save_data(self):
        ensure_directory(DATA_DIR)
        with open(LEARNED_DATA_FILE, 'wb') as f:
            pickle.dump({'samples': self.samples, 'labels': self.labels}, f)

    def compute_features(self, face_img):
        resized = cv2.resize(face_img, (64, 128))
        hist = self.hog.compute(resized)
        return hist.flatten()

    def add_sample(self, face_img, label_int):
        features = self.compute_features(face_img)
        self.samples.append(features)
        self.labels.append(label_int)
        self.train()
        self.save_data()

    def train(self):
        if len(self.samples) < self.k:
            return
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
        prediction = int(results[0][0])
        return prediction, 1.0

    def clear(self):
        self.samples = []
        self.labels = []
        self.trained = False
        self.save_data()

class GenderModel:
    def __init__(self):
        ensure_directory(MODEL_DIR)
        download_file(PROTO_URL, PROTO_FILE)
        download_file(MODEL_URL, MODEL_FILE)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.gender_net = cv2.dnn.readNet(MODEL_FILE, PROTO_FILE)
        self.learner = OnlineLearner()
        self.gender_list = ['Male', 'Female']
        
        self.camera_lock = threading.Lock()
        self.current_frame_jpeg = None
        self.cap = None
        self.last_face_img = None
        self.last_prediction = "Waiting..."
        self.running = True
        self.paused = True # Start in Handshake mode (camera released)
        
        # Start background capture thread
        self.thread = threading.Thread(target=self.update_camera_loop, daemon=True)
        self.thread.start()



    def release_camera(self):
        print("[INFO] Releasing camera for browser handshake...")
        self.paused = True
        
        # Wait up to 3 seconds for the camera to actually be released by the loop
        start = time.time()
        while time.time() - start < 3:
            if self.cap is None:
                print("[INFO] Camera released successfully.")
                return True
            time.sleep(0.1)
            
        print("[WARN] Camera release timed out, forcing...", self.cap)
        return True

    def acquire_camera(self):
        print("[INFO] Re-acquiring camera after handshake...")
        self.paused = False
        return True

    def update_camera_loop(self):
        # Try Initialize Camera
        while self.running:
            if self.paused:
                if self.cap is not None:
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
                time.sleep(0.1)
                continue


            if self.cap is None or not self.cap.isOpened():
                self._open_camera()
                if not self.cap or not self.cap.isOpened():
                    time.sleep(2) # Wait before retry
                    continue



            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                continue
                
            # Copy frame for processing
            frame = cv2.flip(frame, 1)
            frame = frame.copy()
            
            # --- Processing ---
            if not hasattr(self, 'frame_counter'):
                self.frame_counter = 0
            self.frame_counter += 1
            cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            target_rect = None
            max_area = 0
            current_face_detected = False

            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    target_rect = (x, y, w, h)

                face_img = frame[y:y+h, x:x+w].copy()
                learned_label, confidence = self.learner.predict(face_img)
                
                source = "AI"
                color = (255, 0, 0)
                if learned_label is not None:
                    gender = self.gender_list[learned_label]
                    source = "Learned"
                    color = (0, 255, 0)
                else:
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    self.gender_net.setInput(blob)
                    preds = self.gender_net.forward()
                    i = preds[0].argmax()
                    gender = self.gender_list[i]
                
                label = f"{gender}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if target_rect == (x,y,w,h):
                    self.last_prediction = f"{gender} ({source})"
                    current_face_detected = True

            with self.camera_lock:
                if target_rect:
                    x, y, w, h = target_rect
                    self.last_face_img = frame[y:y+h, x:x+w].copy()
                elif not current_face_detected:
                    self.last_prediction = "No Face"
                    
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    self.current_frame_jpeg = jpeg.tobytes()
            
            # Limit Framerate slightly to spare CPU
            time.sleep(0.01) 

    def _open_camera(self):
        # Multi-backend search
        search_configs = [(0, cv2.CAP_ANY), (0, cv2.CAP_DSHOW), (1, cv2.CAP_ANY), (1, cv2.CAP_DSHOW)]
        
        for idx, backend in search_configs:
            print(f"Trying camera {idx}...")
            temp = cv2.VideoCapture(idx, backend)
            if temp.isOpened():
                ret, _ = temp.read()
                if ret:
                    self.cap = temp
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("Camera Opened!")
                    return
                temp.release()
        print("Camera initialization failed.")

    def get_frame(self):
        with self.camera_lock:
            if self.current_frame_jpeg:
                return self.current_frame_jpeg
        
        # Return fallback error image if no frame available
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "CAMERA STANDBY", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def train_last_face(self, gender_label):
        with self.camera_lock:
            if self.last_face_img is None:
                return False, "No face captured"
            target_face = self.last_face_img.copy()
            
        label_int = 0 if gender_label == 'Male' else 1
        self.learner.add_sample(target_face, label_int)
        return True, f"Learned: {gender_label}"

    def clear_memory(self):
        self.learner.clear()
        return True, "Memory Cleared"
