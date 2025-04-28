import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os
import time
import argparse
from collections import deque
from sklearn.preprocessing import StandardScaler

# Try to import YOLO, but make it optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
class GestureRecognitionSystem:
    """
    Comprehensive gesture recognition system capable of recognizing 50+ gestures
    using a combination of techniques:
    1. Static gestures (hand poses) using transfer learning approach
    2. Dynamic gestures (hand movements) using LSTM sequence model
    3. Enhanced landmark detection for more accurate gesture detection
    """
    
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models (set to None initially)
        self.static_gesture_model = None
        self.dynamic_gesture_model = None
        self.custom_landmark_model = None
        
        # YOLO model for person detection (optional)
        self.use_yolo = False
        self.yolo_model = None
        
        # Gesture dictionaries
        self.static_gestures = {}
        self.dynamic_gestures = {}
        
        # Feature scaler for static gestures
        self.scaler = None
        
        # For dynamic gesture recognition
        self.sequence_length = 30
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # For tracking and smoothing predictions
        self.static_history = deque(maxlen=5)
        self.dynamic_history = deque(maxlen=5)
        
        # Directory for models and data
        self.model_dir = "gesture_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load available models
        self.load_models()
        
        # Define 50+ gestures (combined static and dynamic)
        self.define_gestures()
    
    def load_models(self):
        """Load all available trained models."""
        print("Loading models...")
        
        # Try to load static gesture model (transfer learning approach)
        try:
            model_path = os.path.join(self.model_dir, 'static_gesture_model.h5')
            if os.path.exists(model_path):
                self.static_gesture_model = load_model(model_path)
                print("Static gesture model loaded successfully")
                
                # Load gesture names dictionary
                with open(os.path.join(self.model_dir, 'static_gesture_names.pkl'), 'rb') as f:
                    self.static_gestures = pickle.load(f)
                print(f"Loaded {len(self.static_gestures)} static gestures")
                
                # Load feature scaler if available
                try:
                    with open(os.path.join(self.model_dir, 'static_gesture_scaler.pkl'), 'rb') as f:
                        self.scaler = pickle.load(f)
                    print("Static gesture scaler loaded")
                except:
                    print("No scaler found for static gestures")
        except Exception as e:
            print(f"Could not load static gesture model: {e}")
            # Try to load feature-based model as backup
            try:
                model_path = os.path.join(self.model_dir, 'static_gesture_rf_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.static_gesture_model = pickle.load(f)
                    print("Static gesture RF model loaded as backup")
                    
                    # Load gesture names dictionary
                    with open(os.path.join(self.model_dir, 'static_gesture_names.pkl'), 'rb') as f:
                        self.static_gestures = pickle.load(f)
                    print(f"Loaded {len(self.static_gestures)} static gestures")
                    
                    # Load feature scaler
                    with open(os.path.join(self.model_dir, 'static_gesture_scaler.pkl'), 'rb') as f:
                        self.scaler = pickle.load(f)
            except Exception as e:
                print(f"Could not load static gesture RF model: {e}")
        
        # Try to load dynamic gesture model (LSTM)
        try:
            model_path = os.path.join(self.model_dir, 'dynamic_gesture_model.h5')
            if os.path.exists(model_path):
                self.dynamic_gesture_model = load_model(model_path)
                print("Dynamic gesture model loaded successfully")
                
                # Load dynamic gesture names
                with open(os.path.join(self.model_dir, 'dynamic_gesture_names.pkl'), 'rb') as f:
                    self.dynamic_gestures = pickle.load(f)
                print(f"Loaded {len(self.dynamic_gestures)} dynamic gestures")
        except Exception as e:
            print(f"Could not load dynamic gesture model: {e}")
        
        # Try to load custom landmark model
        try:
            model_path = os.path.join(self.model_dir, 'custom_landmark_model.h5')
            if os.path.exists(model_path):
                self.custom_landmark_model = load_model(model_path)
                print("Custom landmark model loaded successfully")
        except Exception as e:
            print(f"Could not load custom landmark model: {e}")
        
        # Initialize YOLO if requested
        if self.use_yolo and YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"Could not load YOLO model: {e}")
                self.use_yolo = False
        elif self.use_yolo:
            print("YOLO was requested but ultralytics package is not installed")
            self.use_yolo = False
    
    def define_gestures(self):
        """Define gestures if not loaded from saved models."""
        # Define static gestures if not already loaded
        if not self.static_gestures:
            self.static_gestures = {
                0: "Open Palm",
                1: "Fist",
                2: "Thumbs Up",
                3: "Peace",
                4: "Pointing",
                5: "OK Sign",
                6: "Pinch",
                7: "Rock Sign",
                8: "Thumbs Down",
                9: "Three Fingers",
                10: "Four Fingers",
                11: "One Finger",
                12: "Gun Shape",
                13: "Phone Call",
                14: "Vulcan Salute",
                15: "Shaka Sign",
                16: "Italian Hand",
                17: "Finger Purse",
                18: "Hand Slice",
                19: "L Shape",
                20: "Claw",
                21: "Finger Crossed",
                22: "Money Rub",
                23: "Finger Snap Position",
                24: "Stop Sign",
                25: "Pinky Promise",
                26: "Finger Guns",
                27: "Index-Middle Connect",
                28: "OK Inverted",
                29: "Writing Pose",
                # Add more to reach 50+ combined with dynamic gestures
            }
        
        # Define dynamic gestures if not already loaded
        if not self.dynamic_gestures:
            self.dynamic_gestures = {
                0: "Swipe Right",
                1: "Swipe Left",
                2: "Swipe Up",
                3: "Swipe Down",
                4: "Wave",
                5: "Circle CW",
                6: "Circle CCW",
                7: "Zoom In",
                8: "Zoom Out",
                9: "Snap",
                10: "Grab",
                11: "Release",
                12: "Push",
                13: "Pull",
                14: "Tap",
                15: "Double Tap",
                16: "Two Finger Scroll",
                17: "Draw Square",
                18: "Draw Triangle",
                19: "Draw X",
                20: "Draw Check Mark",
                # Add more as needed
            }
    
    def generate_hand_image(self, landmarks, img_size=224):
        """Convert hand landmarks to an image representation for CNN input."""
        # Create a blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Draw the hand skeleton on the image
        h, w = img.shape[:2]
        connections = self.mp_hands.HAND_CONNECTIONS
        
        # Scale landmarks to image dimensions
        scaled_landmarks = []
        for landmark in landmarks:
            x_px = min(w - 1, max(0, int(landmark.x * w)))
            y_px = min(h - 1, max(0, int(landmark.y * h)))
            scaled_landmarks.append((x_px, y_px))
        
        # Draw the joints
        for landmark_idx, (x, y) in enumerate(scaled_landmarks):
            # Use different colors for different finger landmarks
            if landmark_idx <= 4:  # Thumb
                color = (255, 0, 0)  # Red
            elif landmark_idx <= 8:  # Index finger
                color = (0, 255, 0)  # Green
            elif landmark_idx <= 12:  # Middle finger
                color = (0, 0, 255)  # Blue
            elif landmark_idx <= 16:  # Ring finger
                color = (255, 255, 0)  # Yellow
            else:  # Pinky
                color = (0, 255, 255)  # Cyan
                
            # Draw the landmark as a circle
            cv2.circle(img, (x, y), 6, color, -1)
        
        # Draw the connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = scaled_landmarks[start_idx]
            end_point = scaled_landmarks[end_idx]
            
            # Determine line color based on which finger
            if start_idx <= 4 or end_idx <= 4:  # Thumb
                color = (255, 0, 0)
            elif start_idx <= 8 or end_idx <= 8:  # Index finger
                color = (0, 255, 0)
            elif start_idx <= 12 or end_idx <= 12:  # Middle finger
                color = (0, 0, 255)
            elif start_idx <= 16 or end_idx <= 16:  # Ring finger
                color = (255, 255, 0)
            else:  # Pinky
                color = (0, 255, 255)
                
            cv2.line(img, start_point, end_point, color, 2)
        
        # Add depth information by varying brightness based on z coordinate
        for landmark_idx, landmark in enumerate(landmarks):
            x_px = min(w - 1, max(0, int(landmark.x * w)))
            y_px = min(h - 1, max(0, int(landmark.y * h)))
            
            # Normalize z to 0-255 range (assuming z is typically in a small range around 0)
            z_val = int((landmark.z + 0.5) * 255) % 256
            
            # Create depth effect with brightness
            cv2.circle(img, (x_px, y_px), 8, (z_val, z_val, z_val), 1)
        
        return img
    
    def extract_hand_features(self, hand_landmarks):
        """Extract features from hand landmarks for static gesture recognition."""
        features = []
        
        # Extract coordinates of all landmarks
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate distances between key landmarks
        for i in range(len(hand_landmarks.landmark)):
            for j in range(i + 1, len(hand_landmarks.landmark)):
                lm1 = hand_landmarks.landmark[i]
                lm2 = hand_landmarks.landmark[j]
                # Euclidean distance
                dist = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
                features.append(dist)
        
        # Add angles between finger joints for better gesture discrimination
        # Thumb
        angles = self.calculate_finger_angles(hand_landmarks)
        features.extend(angles)
        
        return features
    
    def calculate_finger_angles(self, hand_landmarks):
        """Calculate angles between finger joints for better gesture recognition."""
        angles = []
        
        # Define finger joint connections for angle calculation
        finger_connections = [
            # Thumb
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            # Index
            [(0, 5), (5, 6), (6, 7), (7, 8)],
            # Middle
            [(0, 9), (9, 10), (10, 11), (11, 12)],
            # Ring
            [(0, 13), (13, 14), (14, 15), (15, 16)],
            # Pinky
            [(0, 17), (17, 18), (18, 19), (19, 20)]
        ]
        
        landmarks = hand_landmarks.landmark
        
        # Calculate angles for each finger
        for finger in finger_connections:
            for i in range(len(finger) - 2):
                # Get three points to calculate angle
                connection1 = finger[i]
                connection2 = finger[i + 1]
                connection3 = finger[i + 2]
                
                # Extract points
                p1 = np.array([landmarks[connection1[1]].x, landmarks[connection1[1]].y, landmarks[connection1[1]].z])
                p2 = np.array([landmarks[connection2[1]].x, landmarks[connection2[1]].y, landmarks[connection2[1]].z])
                p3 = np.array([landmarks[connection3[1]].x, landmarks[connection3[1]].y, landmarks[connection3[1]].z])
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle
                cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                
                angles.append(angle)
        
        return angles
    
    def detect_static_gesture(self, hand_landmarks):
        """Detect static gesture using the loaded model."""
        if self.static_gesture_model is None:
            return None, 0.0
        
        # Extract features
        features = self.extract_hand_features(hand_landmarks)
        
        # Use transfer learning approach if available
        if hasattr(self.static_gesture_model, 'predict') and \
           not isinstance(self.static_gesture_model, type) and \
           hasattr(self.static_gesture_model, 'layers'):
            # This is a CNN-based model (transfer learning)
            hand_img = self.generate_hand_image(hand_landmarks.landmark)
            img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB) / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = self.static_gesture_model.predict(img, verbose=0)[0]
        else:
            # This is a feature-based model
            # Scale features if scaler is available
            if self.scaler:
                features = self.scaler.transform([features])
            else:
                features = np.array([features])
            
            # Predict
            if hasattr(self.static_gesture_model, 'predict_proba'):
                prediction = self.static_gesture_model.predict_proba(features)[0]
            else:
                prediction = self.static_gesture_model.predict(features, verbose=0)[0]
        
        # Get the gesture ID and confidence
        if isinstance(prediction, np.ndarray) and len(prediction.shape) > 0:
            gesture_id = np.argmax(prediction)
            confidence = prediction[gesture_id]
        else:
            gesture_id = int(prediction)
            confidence = 1.0  # No confidence score available
        
        return gesture_id, confidence
    
    def detect_dynamic_gesture(self):
        """Detect dynamic gesture using sequence data."""
        if self.dynamic_gesture_model is None or len(self.sequence_buffer) < self.sequence_length:
            return None, 0.0
        
        # Convert sequence to numpy array
        sequence = np.array(list(self.sequence_buffer))
        
        # Reshape for model input
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        prediction = self.dynamic_gesture_model.predict(sequence, verbose=0)[0]
        
        # Get the gesture ID and confidence
        gesture_id = np.argmax(prediction)
        confidence = prediction[gesture_id]
        
        return gesture_id, confidence
    
    def process_frame(self, frame):
        """Process a frame for hand gesture recognition."""
        # Create a copy of the frame
        display_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Person detection with YOLO (optional)
        if self.use_yolo and self.yolo_model:
            results = self.yolo_model(frame, classes=[0])  # Class 0 is person
            
            # Draw YOLO detections
            for result in results:
                for box in result.boxes:
                    if box.conf[0] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Process hands with MediaPipe
        hand_results = self.hands.process(rgb_frame)
        
        static_gesture_detected = None
        dynamic_gesture_detected = None
        
        # If hands detected
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Static gesture recognition
                if self.static_gesture_model:
                    gesture_id, confidence = self.detect_static_gesture(hand_landmarks)
                    
                    if gesture_id is not None and confidence > 0.6:
                        # Add to history for smoothing
                        self.static_history.append(gesture_id)
                        
                        # Simple majority voting for smoothing
                        if len(self.static_history) > 0:
                            gesture_counts = {}
                            for g_id in self.static_history:
                                gesture_counts[g_id] = gesture_counts.get(g_id, 0) + 1
                            
                            smoothed_gesture_id = max(gesture_counts, key=gesture_counts.get)
                            static_gesture_detected = self.static_gestures.get(smoothed_gesture_id, "Unknown")
                            
                            # Display static gesture
                            label = f"Static: {static_gesture_detected} ({confidence:.2f})"
                            h, w = display_frame.shape[:2]
                            x_pos = int(min(lm.x * w for lm in hand_landmarks.landmark))
                            y_pos = int(min(lm.y * h for lm in hand_landmarks.landmark)) - 30
                            cv2.putText(display_frame, label, (x_pos, y_pos), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Extract keypoints for dynamic gesture recognition
                if self.dynamic_gesture_model and hand_idx == 0:  # Use only the first hand for dynamic gestures
                    keypoints = []
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Add to sequence buffer
                    self.sequence_buffer.append(keypoints)
                    
                    # Once we have enough frames, detect dynamic gesture
                    if len(self.sequence_buffer) == self.sequence_length:
                        gesture_id, confidence = self.detect_dynamic_gesture()
                        
                        if gesture_id is not None and confidence > 0.6:
                            # Add to history for smoothing
                            self.dynamic_history.append(gesture_id)
                            
                            # Majority voting for smoothing
                            if len(self.dynamic_history) > 0:
                                gesture_counts = {}
                                for g_id in self.dynamic_history:
                                    gesture_counts[g_id] = gesture_counts.get(g_id, 0) + 1
                                
                                smoothed_gesture_id = max(gesture_counts, key=gesture_counts.get)
                                dynamic_gesture_detected = self.dynamic_gestures.get(smoothed_gesture_id, "Unknown")
                                
                                # Display dynamic gesture
                                cv2.putText(display_frame, f"Dynamic: {dynamic_gesture_detected} ({confidence:.2f})", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return display_frame, static_gesture_detected, dynamic_gesture_detected
    
    def collect_gesture_data(self, output_dir="collected_gestures", num_samples=100):
        """Collect data for training the gesture recognition models."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine static and dynamic gestures
        all_gestures = {**self.static_gestures, **{k+100: v for k, v in self.dynamic_gestures.items()}}
        
        # Save gesture names
        with open(os.path.join(output_dir, 'all_gesture_names.pkl'), 'wb') as f:
            pickle.dump(all_gestures, f)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        for gesture_id, gesture_name in all_gestures.items():
            # Create gesture directory
            gesture_dir = os.path.join(output_dir, f"{gesture_id:03d}_{gesture_name.replace(' ', '_')}")
            os.makedirs(gesture_dir, exist_ok=True)
            
            print(f"Prepare to collect data for: {gesture_name}")
            print("Press 's' to start collecting samples")
            
            # Determine if this is a static or dynamic gesture
            is_dynamic = gesture_id >= 100
            
            if is_dynamic:
                print(f"This is a DYNAMIC gesture. You'll need to perform the {gesture_name} motion.")
                print(f"Will collect {num_samples // 2} sequences.")
            else:
                print(f"This is a STATIC gesture. Hold the {gesture_name} pose.")
                print(f"Will collect {num_samples} samples.")
            
            collecting = False
            sample_count = 0
            sequence_data = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands
                hand_results = self.hands.process(rgb_frame)
                
                # Draw landmarks if hands detected
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Display instructions
                cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not collecting:
                    cv2.putText(frame, "Press 's' to start collecting", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    if is_dynamic:
                        cv2.putText(frame, f"Collecting sequence: {sample_count+1}/{num_samples//2}", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"Frame {len(sequence_data)}/{self.sequence_length}", (50, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Collecting: {sample_count}/{num_samples}", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    collecting = True
                    print("Started collecting...")
                
                # Collect data
                if collecting and hand_results.multi_hand_landmarks:
                    if is_dynamic:
                        # For dynamic gestures, collect sequences
                        if len(hand_results.multi_hand_landmarks) > 0:
                            # Extract keypoints from first hand
                            hand_landmarks = hand_results.multi_hand_landmarks[0]
                            keypoints = []
                            for landmark in hand_landmarks.landmark:
                                keypoints.extend([landmark.x, landmark.y, landmark.z])
                            
                            sequence_data.append(keypoints)
                            
                            # Once we have a complete sequence, save it
                            if len(sequence_data) >= self.sequence_length:
                                # Save sequence
                                sequence_file = os.path.join(gesture_dir, f"sequence_{sample_count:03d}.npy")
                                np.save(sequence_file, np.array(sequence_data))
                                
                                sample_count += 1
                                sequence_data = []  # Reset for next sequence
                                print(f"Saved sequence {sample_count}/{num_samples//2}")
                                
                                if sample_count >= num_samples // 2:
                                    break
                    else:
                        # For static gestures, collect individual frames
                        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                            # Extract features
                            features = self.extract_hand_features(hand_landmarks)
                            
                            # Generate hand image
                            hand_img = self.generate_hand_image(hand_landmarks.landmark)
                            
                            # Save feature vector
                            feature_file = os.path.join(gesture_dir, f"features_{sample_count:03d}_hand_{hand_idx}.npy")
                            np.save(feature_file, np.array(features))
                            
                            # Save hand image
                            img_file = os.path.join(gesture_dir, f"image_{sample_count:03d}_hand_{hand_idx}.png")
                            cv2.imwrite(img_file, hand_img)
                            
                            sample_count += 1
                            print(f"Saved sample {sample_count}/{num_samples}")
                            
                            if sample_count >= num_samples:
                                break
                            
                            # Brief delay to avoid duplicates
                            time.sleep(0.1)
                
                if (is_dynamic and sample_count >= num_samples // 2) or \
                   (not is_dynamic and sample_count >= num_samples):
                    break
            
            print(f"Completed collecting data for {gesture_name}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection complete!")
    
    def train_models(self, data_dir="collected_gestures"):
        """Train all gesture recognition models."""
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} not found.")
            return
        
        # Load gesture names
        try:
            with open(os.path.join(data_dir, 'all_gesture_names.pkl'), 'rb') as f:
                all_gestures = pickle.load(f)
            
            # Split into static and dynamic gestures
            self.static_gestures = {k: v for k, v in all_gestures.items() if k < 100}
            self.dynamic_gestures = {k-100: v for k, v in all_gestures.items() if k >= 100}
            
            # Save the dictionaries
            with open(os.path.join(self.model_dir, 'static_gesture_names.pkl'), 'wb') as f:
                pickle.dump(self.static_gestures, f)
            
            with open(os.path.join(self.model_dir, 'dynamic_gesture_names.pkl'), 'wb') as f:
                pickle.dump(self.dynamic_gestures, f)
        except Exception as e:
            print(f"Error loading gesture names: {e}")
            return
        
        # Train static gesture model using transfer learning
        if self.static_gestures:
            self.train_static_model(data_dir)
        
        # Train dynamic gesture model using LSTM
        if self.dynamic_gestures:
            self.train_dynamic_model(data_dir)
    
    def train_static_model(self, data_dir):
        """Train static gesture model using transfer learning."""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        
        print("Training static gesture model...")
        
        # Load images and features
        images = []
        features = []
        labels = []
        
        # Get all static gesture directories
        gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and 
                        int(d.split('_')[0]) < 100]
        
        for dir_name in gesture_dirs:
            gesture_id = int(dir_name.split('_')[0])
            gesture_dir = os.path.join(data_dir, dir_name)
            
            # Get image files
            img_files = [f for f in os.listdir(gesture_dir) if f.startswith('image_') and f.endswith('.png')]
            
            for img_file in img_files:
                # Load image
                img_path = os.path.join(gesture_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load corresponding feature file
                feature_file = img_file.replace('image_', 'features_').replace('.png', '.npy')
                feature_path = os.path.join(gesture_dir, feature_file)
                
                if os.path.exists(feature_path):
                    feature_vector = np.load(feature_path)
                    
                    # Add to datasets
                    images.append(img)
                    features.append(feature_vector)
                    labels.append(gesture_id)
        
        # Convert to numpy arrays
        X_img = np.array(images) / 255.0  # Normalize images
        X_features = np.array(features)
        y = to_categorical(np.array(labels))
        
        print(f"Loaded {len(X_img)} samples for {len(self.static_gestures)} static gestures")
        
        # Create feature scaler
        self.scaler = StandardScaler()
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        # Save scaler
        with open(os.path.join(self.model_dir, 'static_gesture_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            X_img, X_features_scaled, y, test_size=0.2, random_state=42)
        
        # Create transfer learning model
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.static_gestures), activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'static_gesture_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_img_train, y_train,
            validation_data=(X_img_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save model
        model.save(os.path.join(self.model_dir, 'static_gesture_model.h5'))
        
        # Also train a feature-based model as backup
        from sklearn.ensemble import RandomForestClassifier
        print("Training feature-based model as backup...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_feat_train, np.argmax(y_train, axis=1))
        
        # Save feature-based model
        with open(os.path.join(self.model_dir, 'static_gesture_rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        
        print("Static gesture models trained and saved successfully!")
        
        # Update class model reference
        self.static_gesture_model = model
    
    def train_dynamic_model(self, data_dir):
        """Train dynamic gesture model using LSTM."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        
        print("Training dynamic gesture model...")
        
        # Load sequence data
        sequences = []
        labels = []
        
        # Get all dynamic gesture directories
        gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and 
                       int(d.split('_')[0]) >= 100]
        
        for dir_name in gesture_dirs:
            gesture_id = int(dir_name.split('_')[0]) - 100  # Adjust ID to start from 0
            gesture_dir = os.path.join(data_dir, dir_name)
            
            # Get sequence files
            seq_files = [f for f in os.listdir(gesture_dir) if f.startswith('sequence_') and f.endswith('.npy')]
            
            for seq_file in seq_files:
                # Load sequence
                seq_path = os.path.join(gesture_dir, seq_file)
                sequence = np.load(seq_path)
                
                sequences.append(sequence)
                labels.append(gesture_id)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = to_categorical(np.array(labels))
        
        print(f"Loaded {len(X)} sequences for {len(self.dynamic_gestures)} dynamic gestures")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(len(self.dynamic_gestures), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'dynamic_gesture_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save model
        model.save(os.path.join(self.model_dir, 'dynamic_gesture_model.h5'))
        
        print("Dynamic gesture model trained and saved successfully!")
        
        # Update class model reference
        self.dynamic_gesture_model = model
    
    def run(self):
        """Run the gesture recognition system on webcam feed."""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # For FPS calculation
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, static_gesture, dynamic_gesture = self.process_frame(frame)
            
            # Display FPS
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show help text
            cv2.putText(processed_frame, "Press 'q' to quit", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show result
            cv2.imshow('50+ Gesture Recognition System', processed_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the gesture recognition system."""
    parser = argparse.ArgumentParser(description='50+ Gesture Recognition System')
    parser.add_argument('--collect', action='store_true', help='Collect training data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--run', action='store_true', help='Run the recognition system')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per gesture to collect')
    parser.add_argument('--yolo', action='store_true', help='Use YOLO for person detection')
    
    args = parser.parse_args()
    
    # Create the system
    system = GestureRecognitionSystem()
    system.use_yolo = args.yolo
    
    if args.collect:
        system.collect_gesture_data(num_samples=args.samples)
    elif args.train:
        system.train_models()
    elif args.run:
        system.run()
    else:
        # Default to run mode if no option specified
        system.run()

if __name__ == "__main__":
    main()