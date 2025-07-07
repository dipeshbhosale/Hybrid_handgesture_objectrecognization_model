import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys
import time
import os
import joblib
from collections import deque
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Import hybrid ultimate detection system
try:
    from hybrid_ultimate_detection import HybridUltimateDetectionSystem, run_hybrid_ultimate_detection_system
    from datetime import datetime
    HYBRID_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Hybrid detection system not available: {e}")
    HYBRID_DETECTION_AVAILABLE = False

# Optional: for keyboard capture during data collection
try:
    import keyboard
except ImportError:
    keyboard = None

# --- INTEGRATED COLLECTION FUNCTION ---
def collect_gesture_data_integrated():
    """
    Integrated gesture collection with multiple gestures in one session.
    """
    if not keyboard:
        print("❌ 'keyboard' package required for data collection. Install with: pip install keyboard")
        return
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    all_data = []
    all_labels = []
    gestures = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
    
    for gesture in gestures:
        print(f"\n=== Collecting data for gesture: {gesture} ===")
        print("Position your hand and press SPACE to capture samples.")
        print("Collect at least 30 samples per gesture. Press 'q' to move to next gesture.\n")
        
        cap = cv2.VideoCapture(0)
        gesture_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {gesture_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Next gesture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    if keyboard.is_pressed("space") and len(landmarks) == 63:
                        all_data.append(landmarks)
                        all_labels.append(gesture)
                        gesture_count += 1
                        print(f"Captured {gesture} sample #{gesture_count}")
            
            cv2.imshow("Gesture Collection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        print(f"Collected {gesture_count} samples for {gesture}")
    
    cv2.destroyAllWindows()
    
    if all_data:
        df = pd.DataFrame(all_data)
        df["label"] = all_labels
        df.to_csv("gesture_data.csv", index=False)
        print(f"\n✅ Saved {len(all_data)} total samples to gesture_data.csv")
        return True
    return False

# --- IMPROVED TRAINING FUNCTION ---
def augment_jitter(X, y, jitter_std=0.01, n_aug=2):
    """
    Augment data by adding Gaussian noise (jitter) to landmark features.
    """
    X_aug = []
    y_aug = []
    for xi, yi in zip(X, y):
        for _ in range(n_aug):
            noise = np.random.normal(0, jitter_std, size=xi.shape)
            X_aug.append(xi + noise)
            y_aug.append(yi)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    return np.vstack([X, X_aug]), np.hstack([y, y_aug])

def advanced_data_augmentation(X, y, augment_factor=3):
    """
    Advanced data augmentation with rotation, scaling, and multiple noise types.
    """
    X_aug = []
    y_aug = []
    
    for xi, yi in zip(X, y):
        # Original sample
        X_aug.append(xi)
        y_aug.append(yi)
        
        for _ in range(augment_factor):
            # Reshape to 21 landmarks × 3 coordinates
            landmarks = xi.reshape(21, 3)
            
            # 1. Gaussian noise
            noise = np.random.normal(0, 0.005, landmarks.shape)
            noisy_landmarks = landmarks + noise
            
            # 2. Scaling augmentation
            scale_factor = np.random.uniform(0.95, 1.05)
            scaled_landmarks = landmarks * scale_factor
            
            # 3. Translation (small shifts)
            translation = np.random.normal(0, 0.01, (1, 3))
            translated_landmarks = landmarks + translation
            
            # 4. Coordinate dropout (randomly set some coords to mean)
            dropout_landmarks = landmarks.copy()
            if np.random.random() > 0.7:
                dropout_idx = np.random.choice(21, size=3, replace=False)
                dropout_landmarks[dropout_idx] = np.mean(landmarks, axis=0)
            
            # Add augmented samples
            for aug_landmarks in [noisy_landmarks, scaled_landmarks, translated_landmarks, dropout_landmarks]:
                # Ensure landmarks are within valid range [0, 1]
                aug_landmarks = np.clip(aug_landmarks, 0, 1)
                X_aug.append(aug_landmarks.flatten())
                y_aug.append(yi)
    
    return np.array(X_aug), np.array(y_aug)

def train_model_advanced(csv_path="gesture_data.csv", model_path="gesture_model.pkl"):
    """
    Advanced training with ensemble methods, feature selection, and robust validation.
    """
    if not os.path.exists(csv_path):
        print("❌ gesture_data.csv not found! Run data collection first.")
        return False

    print("📊 Loading gesture data...")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns or df.shape[1] != 64:
        print(f"❌ Invalid CSV format. Expected 64 columns, got {df.shape[1]}")
        return False

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"Original data: {X.shape[0]} samples, {len(np.unique(y))} classes")

    # Advanced data augmentation
    print("🔄 Applying advanced data augmentation...")
    X_aug, y_aug = advanced_data_augmentation(X, y, augment_factor=2)
    print(f"Augmented data: {X_aug.shape[0]} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    print("🎯 Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(45, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Define multiple models with regularization
    models = {
        'SVC': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Grid search parameters
    param_grids = {
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        },
        'GradientBoosting': {
            'n_estimators': [100, 150],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    }

    # Train and evaluate models
    best_models = {}
    cv_scores = {}
    
    # Stratified K-Fold for robust validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n🔍 Training {name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=3, n_jobs=-1, scoring='accuracy',
            verbose=0
        )
        grid_search.fit(X_train_selected, y_train)
        
        best_models[name] = grid_search.best_estimator_
        
        # Cross-validation on best model
        cv_score = cross_val_score(
            best_models[name], X_train_selected, y_train, 
            cv=skf, scoring='accuracy'
        )
        cv_scores[name] = cv_score
        
        test_score = best_models[name].score(X_test_selected, y_test)
        
        print(f"{name} - Best params: {grid_search.best_params_}")
        print(f"{name} - CV Score: {cv_score.mean():.3f} (±{cv_score.std()*2:.3f})")
        print(f"{name} - Test Score: {test_score:.3f}")

    # Create ensemble model
    print("\n🎭 Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate ensemble
    ensemble_cv = cross_val_score(ensemble, X_train_selected, y_train, cv=skf, scoring='accuracy')
    ensemble_test = ensemble.score(X_test_selected, y_test)
    
    print(f"Ensemble - CV Score: {ensemble_cv.mean():.3f} (±{ensemble_cv.std()*2:.3f})")
    print(f"Ensemble - Test Score: {ensemble_test:.3f}")

    # Select best model
    all_scores = {name: best_models[name].score(X_test_selected, y_test) for name in best_models}
    all_scores['Ensemble'] = ensemble_test
    
    best_model_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_model_name]
    
    if best_model_name == 'Ensemble':
        final_model = ensemble
    else:
        final_model = best_models[best_model_name]
    
    # Create final pipeline with preprocessing
    from sklearn.pipeline import Pipeline
    final_pipeline = Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('classifier', final_model)
    ])
    
    # Refit on original training data
    final_pipeline.fit(X_train, y_train)
    final_test_score = final_pipeline.score(X_test, y_test)

    print(f"\n✅ Best model: {best_model_name}")
    print(f"🎯 Final test accuracy: {final_test_score:.3f}")
    
    # Check for overfitting
    if best_model_name in cv_scores:
        cv_mean = cv_scores[best_model_name].mean()
        if final_test_score - cv_mean > 0.1:
            print("⚠️  Warning: Possible overfitting detected!")
        else:
            print("✅ Model appears to generalize well.")

    joblib.dump(final_pipeline, model_path)
    print(f"✅ Model pipeline saved as {model_path}")
    
    return True

# --- 3️⃣ Ensure gesture_model.pkl exists or auto-train if missing ---
def ensure_model_exists(csv_path="gesture_data.csv", model_path="gesture_model.pkl"):
    """
    Ensure gesture_model.pkl exists. If not, train and save from gesture_data.csv if available.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        if os.path.exists(csv_path):
            print(f"Found '{csv_path}'. Training model now...")
            train_model_advanced()
        else:
            raise FileNotFoundError(
                f"Neither '{model_path}' nor '{csv_path}' found. Please collect data and train the model first."
            )

# --- 3️⃣ REAL-TIME PREDICTION (OpenCV version) ---
def predict_gesture_realtime(model_path="gesture_model.pkl"):
    """
    Real-time webcam gesture prediction using trained model and MediaPipe.
    Limited to 15 FPS and 360p resolution for better performance.
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        return
    model = joblib.load(model_path)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to 360p and FPS to 15
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Frame rate control
    fps_limit = 15
    frame_time = 1.0 / fps_limit
    last_time = time.time()
    
    print("Press 'q' to quit. Running at 15 FPS @ 360p for optimal performance.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # FPS control
        current_time = time.time()
        if current_time - last_time < frame_time:
            continue
        last_time = current_time
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture_label = "No hand"
        confidence = 0.0
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    features_np = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(features_np)[0]
                    if hasattr(model, "predict_proba"):
                        conf = np.max(model.predict_proba(features_np))
                        confidence = float(conf)
                        gesture_label = f"{pred} {int(confidence*100)}%"
                    else:
                        gesture_label = str(pred)
                else:
                    gesture_label = "Incomplete landmarks"
        
        # Add FPS info to display
        cv2.putText(
            frame, gesture_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "15 FPS @ 360p", (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        cv2.imshow("Real-Time Gesture Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 3️⃣ REAL-TIME PREDICTION (Gradio version) ---
import gradio as gr

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Create persistent MediaPipe Hands object for performance ---
# This avoids re-initializing the model on every frame, which is the main cause of lag.
hands_solution = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0  # Use the lightest model
)

# Optimized FPS control for Gradio processing
last_gradio_time = time.time()
gradio_fps_limit = 15  # Increased back to 15 FPS for smoother video
gradio_frame_time = 1.0 / gradio_fps_limit
frame_skip_counter = 0

def detect_gesture(image):
    """
    Optimized gesture detection for Gradio with 15 FPS for smoother experience.
    Uses a persistent MediaPipe model with reduced frame skipping.
    """
    global last_gradio_time, frame_skip_counter
    
    # Reduced frame skipping for smoother video
    frame_skip_counter += 1
    if frame_skip_counter % 2 != 0:  # Process only every 2nd frame instead of 4th
        if hasattr(detect_gesture, "last_result"):
            return detect_gesture.last_result
        return None, "Processing..."
    
    # FPS control for Gradio
    current_time = time.time()
    if current_time - last_gradio_time < gradio_frame_time:
        if hasattr(detect_gesture, "last_result"):
            return detect_gesture.last_result
        return None, "Processing at 15 FPS..."
    
    last_gradio_time = current_time
    
    if image is None:
        return None, "No input - Check camera permissions"
    
    try:
        frame_rgb = np.array(image)
        if frame_rgb.size == 0:
            return None, "Empty frame - Camera not working"
        
        # Keep reasonable size for gesture detection
        height, width = frame_rgb.shape[:2]
        if width > 1280:  # Only resize if too large
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        elif width < 480:  # Don't go too small
            scale = 480 / width
            new_width = 480
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        gesture_label = "No hand detected"
        confidence = 0.0

        # Lazy load model (only when needed)
        if not hasattr(detect_gesture, "model"):
            try:
                ensure_model_exists()
                model_data = joblib.load("gesture_model.pkl")
                if isinstance(model_data, dict):
                    detect_gesture.model = model_data.get('model') or model_data.get('pipeline')
                else:
                    detect_gesture.model = model_data
                print("✅ Model loaded successfully")
            except Exception as e:
                return frame_rgb, f"Model Error: {str(e)}"

        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr = frame_bgr.copy()

        # Process the frame using the persistent hands object (no 'with' block)
        results = hands_solution.process(frame_rgb)
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lighter landmark drawing
                mp_drawing.draw_landmarks(
                    annotated_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                
                if len(features) == 63:
                    features_np = np.array(features).reshape(1, -1)
                    model = detect_gesture.model
                    
                    try:
                        pred = model.predict(features_np)[0]
                        if hasattr(model, "predict_proba"):
                            conf = np.max(model.predict_proba(features_np))
                            confidence = float(conf)
                            gesture_label = f"{pred} ({int(confidence*100)}%)"
                        else:
                            gesture_label = str(pred)
                    except Exception as e:
                        gesture_label = f"Prediction error"
                else:
                    gesture_label = f"Invalid landmarks: {len(features)}/63"
        
        # Simplified text overlay
        cv2.putText(
            annotated_bgr, gesture_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # Add optimized status
        status_text = "Hand OK" if results.multi_hand_landmarks else "No Hand"
        cv2.putText(
            annotated_bgr, status_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA
        )
        
        # Add performance indicator
        cv2.putText(
            annotated_bgr, "15 FPS (Optimized)", (10, annotated_bgr.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Convert back to RGB for Gradio
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Cache result for performance
        detect_gesture.last_result = (annotated_rgb, gesture_label)
        
        return annotated_rgb, gesture_label
        
    except Exception as e:
        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)[:30]}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(error_frame, "15 FPS (Optimized)", (10, 340),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return error_frame, f"Error: {str(e)[:50]}"

# --- OBJECT DETECTION FUNCTIONS ---

def load_object_detection_imports():
    """
    Load object detection dependencies
    """
    try:
        global YOLO, torch
        from ultralytics import YOLO
        import torch
        return True
    except ImportError as e:
        print(f"⚠️ Object detection not available: {e}")
        return False

def load_yolo_model(model_path="yolov8s.pt"):
    """
    Load enhanced YOLO model with PyTorch compatibility fix and model selection
    """
    try:
        # Try different models in order of preference (better accuracy)
        model_preferences = [
            "yolov8m.pt",    # Medium - good balance
            "yolov8s.pt",    # Small - faster
            "yolov8l.pt",    # Large - high accuracy
            "yolov8n.pt",    # Nano - fastest
        ]
        
        # Add safe globals for ultralytics compatibility
        try:
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules.head.Detect',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.block.SPPF',
                'collections.OrderedDict'
            ])
        except:
            pass
        
        # Fix for PyTorch compatibility
        old_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return old_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load
        
        # Try to load the best available model
        for model_name in model_preferences:
            try:
                print(f"🔄 Attempting to load {model_name}...")
                model = YOLO(model_name)
                torch.load = old_load  # Restore original
                
                # Test the model with a dummy prediction
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = model(dummy_frame, verbose=False)
                
                print(f"✅ Successfully loaded {model_name}")
                print(f"   📊 Classes available: {len(model.names)}")
                print(f"   🎯 Sample classes: {list(model.names.values())[:10]}")
                
                return model
                
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        # Restore original torch.load
        torch.load = old_load
        print("❌ All model loading attempts failed")
        return None
        
    except Exception as e:
        print(f"❌ Critical error loading YOLO model: {e}")
        return None

def detect_objects_in_frame(frame, model, confidence_threshold=0.25):
    """
    Enhanced multi-object detection with improved accuracy and visualization
    """
    if model is None:
        return frame, []
    
    try:
        # Enhanced YOLO inference with optimized parameters for better detection
        results = model(
            frame, 
            conf=confidence_threshold,  # Lower confidence for more detections
            iou=0.4,                   # Lower IOU for better overlapping detection
            max_det=100,               # More detections allowed
            verbose=False,
            save=False,
            show=False,
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        detected_objects = []
        # Enhanced color palette for better object distinction
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green  
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),     # Dark Green
            (128, 128, 0),   # Olive
            (0, 0, 128),     # Navy
            (128, 0, 0),     # Maroon
            (0, 128, 128),   # Teal
            (192, 192, 192), # Silver
            (255, 215, 0),   # Gold
            (255, 69, 0),    # Red Orange
            (50, 205, 50),   # Lime Green
            (30, 144, 255),  # Dodger Blue
            (220, 20, 60)    # Crimson
        ]
        
        object_counts = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Count objects by class
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    detected_objects.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'class_id': class_id,
                        'area': (x2 - x1) * (y2 - y1)
                    })
                    
                    # Enhanced visualization with color coding
                    color = colors[class_id % len(colors)]
                    
                    # Draw thick bounding box with gradient effect
                    thickness = max(2, int(conf * 6))  # Thickness based on confidence
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Add border for better visibility
                    cv2.rectangle(frame, (int(x1)-1, int(y1)-1), (int(x2)+1, int(y2)+1), (255,255,255), 1)
                    
                    # Enhanced label with better styling
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Background rectangle with shadow effect
                    cv2.rectangle(frame, 
                                (int(x1), int(y1) - label_size[1] - 15),
                                (int(x1) + label_size[0] + 10, int(y1) - 5),
                                (0, 0, 0), -1)  # Black shadow
                    cv2.rectangle(frame, 
                                (int(x1), int(y1) - label_size[1] - 12),
                                (int(x1) + label_size[0] + 8, int(y1) - 2),
                                color, -1)  # Colored background
                    
                    # White text for better contrast
                    cv2.putText(frame, label, (int(x1) + 4, int(y1) - 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add object index number for tracking
                    cv2.putText(frame, f"#{i+1}", (int(x1), int(y2) + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add enhanced statistics overlay
        _add_detection_statistics(frame, detected_objects, object_counts)
        
        return frame, detected_objects
        
    except Exception as e:
        print(f"Enhanced detection error: {e}")
        # Add error visualization
        cv2.putText(frame, f"Detection Error: {str(e)[:30]}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, []

def _add_detection_statistics(frame, detected_objects, object_counts):
    """Add enhanced statistics overlay to the detection frame"""
    height, width = frame.shape[:2]
    
    # Statistics panel
    panel_width = 280
    panel_height = min(350, height - 40)
    panel_x = width - panel_width - 10
    panel_y = 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Border
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 255, 255), 2)
    
    # Title
    cv2.putText(frame, "🎯 LIVE DETECTION", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset = panel_y + 50
    
    # Total objects with prominent display
    cv2.putText(frame, f"TOTAL: {len(detected_objects)} objects", 
               (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    
    # Unique types
    cv2.putText(frame, f"TYPES: {len(object_counts)} unique", 
               (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    y_offset += 25
    
    # Separator
    cv2.line(frame, (panel_x + 10, y_offset), (panel_x + panel_width - 10, y_offset),
            (255, 255, 255), 1)
    y_offset += 15
    
    # Object breakdown
    cv2.putText(frame, "DETECTED OBJECTS:", (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    y_offset += 20
    
    # List objects with counts
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (255, 165, 0)]
    
    for i, (class_name, count) in enumerate(sorted(object_counts.items())):
        if y_offset < panel_y + panel_height - 25:
            color = colors[i % len(colors)]
            text = f"• {class_name}: {count}"
            cv2.putText(frame, text, (panel_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 18
    
    # Performance indicator
    cv2.putText(frame, "Enhanced Detection", (panel_x + 10, panel_y + panel_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

def detect_objects_only_opencv():
    """
    Enhanced OpenCV interface for object detection - supports multiple objects with advanced features
    """
    try:
        if not load_object_detection_imports():
            print("❌ Object detection dependencies not available")
            return
        
        model = load_yolo_model()
        if model is None:
            print("❌ Failed to load YOLO model")
            return
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎯 Enhanced Object Detection Mode")
        print("=" * 50)
        print("🚀 Features:")
        print("• Advanced multi-object detection")
        print("• Real-time statistics and tracking")
        print("• Enhanced visualization with color coding")
        print("• Live performance monitoring")
        print("\n🎮 Controls:")
        print("• 'q' - Quit")
        print("• '+/-' - Adjust confidence threshold")
        print("• 's' - Save screenshot")
        print("• 'r' - Reset statistics")
        
        confidence = 0.25  # Lower for better detection
        detection_history = []
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Detect objects with enhanced function
            annotated_frame, detected_objects = detect_objects_in_frame(
                frame, model, confidence_threshold=confidence
            )
            
            # Additional overlay information
            height, width = annotated_frame.shape[:2]
            
            # Header with title and stats
            cv2.putText(annotated_frame, "ENHANCED OBJECT DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # FPS calculation
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - start_time) if start_time > 0 else 0
                start_time = current_time
                
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Detection history tracking
            detection_history.append(len(detected_objects))
            if len(detection_history) > 100:
                detection_history.pop(0)
            
            # Average objects over time
            if detection_history:
                avg_objects = sum(detection_history) / len(detection_history)
                cv2.putText(annotated_frame, f"Avg Objects: {avg_objects:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 2)
            
            # Controls info
            controls_y = height - 60
            cv2.putText(annotated_frame, "Controls: Q=Quit | +/-=Confidence | S=Save | R=Reset", 
                       (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Enhanced Multi-Object Detection", annotated_frame)
            
            # Enhanced keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                confidence = min(0.9, confidence + 0.05)
                print(f"📈 Confidence threshold: {confidence:.2f}")
            elif key == ord('-'):
                confidence = max(0.1, confidence - 0.05)
                print(f"📉 Confidence threshold: {confidence:.2f}")
            elif key == ord('s'):
                screenshot_name = f"detection_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('r'):
                detection_history.clear()
                print("🔄 Statistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print(f"\n📊 Detection Session Summary:")
        print(f"• Total frames processed: {frame_count}")
        if detection_history:
            print(f"• Average objects per frame: {sum(detection_history)/len(detection_history):.1f}")
            print(f"• Maximum objects detected: {max(detection_history)}")
        print(f"• Final confidence threshold: {confidence:.2f}")
        
    except Exception as e:
        print(f"❌ Enhanced object detection error: {e}")
        print(f"💡 Try running: pip install --upgrade ultralytics torch torchvision")

def detect_gesture_and_objects_opencv():
    """
    Combined gesture and object detection using OpenCV - supports multiple objects
    """
    try:
        # Load gesture model
        gesture_model = None
        if os.path.exists("gesture_model.pkl"):
            gesture_model = joblib.load("gesture_model.pkl")
            print("✅ Gesture model loaded")
        else:
            print("⚠️ Gesture model not found - gesture detection disabled")
        
        # Load object detection
        object_model = None
        if load_object_detection_imports():
            object_model = load_yolo_model()
            if object_model:
                print("✅ YOLO model loaded")
            else:
                print("⚠️ YOLO model failed to load - object detection disabled")
        else:
            print("⚠️ Object detection dependencies not available")
        
        if gesture_model is None and object_model is None:
            print("❌ No models available - exiting")
            return
        
        # Initialize MediaPipe for gestures
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("🤖 Combined Gesture + Object Detection")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # --- GESTURE DETECTION ---
            gesture_result = "No hand detected"
            if gesture_model is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Extract features for prediction
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        if len(landmarks) == 63:
                            try:
                                features_np = np.array(landmarks).reshape(1, -1)
                                pred = gesture_model.predict(features_np)[0]
                                
                                if hasattr(gesture_model, "predict_proba"):
                                    conf = np.max(gesture_model.predict_proba(features_np))
                                    gesture_result = f"{pred} ({int(conf*100)}%)"
                                else:
                                    gesture_result = pred
                            except Exception as e:
                                gesture_result = "Processing error"
            
            # --- OBJECT DETECTION ---
            detected_objects = []
            if object_model is not None:
                frame, detected_objects = detect_objects_in_frame(
                    frame, object_model, confidence_threshold=0.5
                )
            
            # --- DISPLAY RESULTS ---
            # Gesture info
            cv2.putText(frame, f"Gesture: {gesture_result}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Object info
            if detected_objects:
                # Group objects by class for display
                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                y_offset = 60
                for class_name, count in object_counts.items():
                    text = f"{class_name}: {count}"
                    cv2.putText(frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                # Total count
                total_text = f"Total Objects: {len(detected_objects)}"
                cv2.putText(frame, total_text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Objects: None detected", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Status info
            gesture_status = "✅" if gesture_model else "❌"
            object_status = "✅" if object_model else "❌"
            cv2.putText(frame, f"Gesture: {gesture_status} | Objects: {object_status}", 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Combined AI Vision", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Combined detection error: {e}")

# --- COMBINED GRADIO INTERFACE FOR GESTURE + OBJECT DETECTION ---

def detect_gesture_and_objects_gradio(image):
    """
    Combined gesture and object detection for Gradio interface
    Processes both hand gestures and object detection simultaneously
    """
    global last_gradio_time, frame_skip_counter
    
    # Performance control - process every 3rd frame for combined detection
    frame_skip_counter += 1
    if frame_skip_counter % 3 != 0:
        if hasattr(detect_gesture_and_objects_gradio, "last_result"):
            return detect_gesture_and_objects_gradio.last_result
        return None, "Processing...", "Processing...", "Loading..."
    
    # FPS control
    current_time = time.time()
    if current_time - last_gradio_time < gradio_frame_time * 2:  # Slower for combined processing
        if hasattr(detect_gesture_and_objects_gradio, "last_result"):
            return detect_gesture_and_objects_gradio.last_result
        return None, "Processing at 8 FPS...", "Optimizing...", "Please wait..."
    
    last_gradio_time = current_time
    
    if image is None:
        return None, "No input - Check camera", "No camera input", "Camera issue"
    
    try:
        frame_rgb = np.array(image)
        if frame_rgb.size == 0:
            return None, "Empty frame", "Camera not working", "Error"
        
        # Keep larger resolution for better multi-object detection in full screen
        height, width = frame_rgb.shape[:2]
        if width > 1280:  # Only resize if too large
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        elif width < 640:  # Don't go too small
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Initialize results
        gesture_result = "No hand detected"
        object_result = "No objects detected"
        stats_result = "Initializing..."
        
        # --- LAZY LOAD MODELS ---
        if not hasattr(detect_gesture_and_objects_gradio, "gesture_model"):
            try:
                ensure_model_exists()
                detect_gesture_and_objects_gradio.gesture_model = joblib.load("gesture_model.pkl")
                print("✅ Gesture model loaded for Gradio")
            except Exception as e:
                detect_gesture_and_objects_gradio.gesture_model = None
                print(f"⚠️ Gesture model not available: {e}")
        
        if not hasattr(detect_gesture_and_objects_gradio, "yolo_model"):
            try:
                if load_object_detection_imports():
                    detect_gesture_and_objects_gradio.yolo_model = load_yolo_model()
                    print("✅ YOLO model loaded for Gradio")
                else:
                    detect_gesture_and_objects_gradio.yolo_model = None
            except Exception as e:
                detect_gesture_and_objects_gradio.yolo_model = None
                print(f"⚠️ YOLO model not available: {e}")
        
        # Convert to BGR for processing
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated_frame = frame_bgr.copy()
        
        # --- GESTURE DETECTION ---
        if detect_gesture_and_objects_gradio.gesture_model is not None:
            results = hands_solution.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Extract features
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    if len(features) == 63:
                        features_np = np.array(features).reshape(1, -1)
                        model = detect_gesture_and_objects_gradio.gesture_model
                        
                        try:
                            pred = model.predict(features_np)[0]
                            if hasattr(model, "predict_proba"):
                                conf = np.max(model.predict_proba(features_np))
                                confidence = float(conf)
                                gesture_result = f"{pred} ({int(confidence*100)}%)"
                            else:
                                gesture_result = str(pred)
                        except Exception as e:
                            gesture_result = f"Prediction error"
                    else:
                        gesture_result = f"Invalid landmarks: {len(features)}/63"
        
        # --- OBJECT DETECTION ---
        detected_objects = []
        if detect_gesture_and_objects_gradio.yolo_model is not None:
            try:
                annotated_frame, detected_objects = detect_objects_in_frame(
                    annotated_frame, detect_gesture_and_objects_gradio.yolo_model, 
                    confidence_threshold=0.6
                )
                
                if detected_objects:
                    object_names = [f"{obj['class']} ({obj['confidence']:.1f})" for obj in detected_objects[:3]]
                    object_result = f"🎯 Objects: {', '.join(object_names)}"
                    if len(detected_objects) > 3:
                        object_result += f" + {len(detected_objects)-3} more"
                else:
                    object_result = "🎯 No objects detected"
            except Exception as e:
                object_result = f"🎯 Object detection error: {str(e)[:30]}"
        else:
            object_result = "🎯 Object detection not available"
        
        # --- STATISTICS ---
        gesture_available = "✅" if detect_gesture_and_objects_gradio.gesture_model else "❌"
        object_available = "✅" if detect_gesture_and_objects_gradio.yolo_model else "❌"
        
        stats_result = f"""📊 AI Status:
• Gesture Recognition: {gesture_available}
• Object Detection: {object_available}  
• Objects Found: {len(detected_objects)}
• Processing: Combined AI @ 8 FPS
• Frame: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}"""
        
        # Add combined overlay to frame
        cv2.putText(annotated_frame, "Combined AI Vision", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, f"G:{gesture_available} O:{object_available}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Cache result
        detect_gesture_and_objects_gradio.last_result = (result_rgb, gesture_result, object_result, stats_result)
        
        return result_rgb, gesture_result, object_result, stats_result
        
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}"
        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Combined AI Error", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(error_frame, error_msg, (10, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return error_frame, error_msg, error_msg, error_msg

# Combined Gradio Interface with Full Screen Support
css = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
}
#component-0 {
    max-width: 100% !important;
}
.block {
    max-width: none !important;
}
"""

with gr.Blocks(title="Combined AI Vision - FULL SCREEN", theme=gr.themes.Soft(), css=css) as combined_iface:
    gr.Markdown("# 🤖 Combined AI Vision: Gestures + Objects - FULL SCREEN MODE")
    gr.Markdown("### Real-time hand gesture recognition and object detection powered by AI - Enhanced for multiple objects")
    
    with gr.Row():
        with gr.Column(scale=4):  # Increased scale for larger display
            # Main webcam feed - FULL SCREEN
            webcam_combined = gr.Image(
                label="📹 Live Combined AI Vision - FULL SCREEN",
                streaming=True,
                height=720,  # Increased height
                width=1280,  # Increased width
                mirror_webcam=False
            )
        
        with gr.Column(scale=2):  # Reduced scale to give more space to video
            # Results panel
            gesture_output = gr.Textbox(
                label="🤟 Gesture Recognition",
                interactive=False,
                placeholder="Detecting hand gestures...",
                lines=3  # Increased lines
            )
            
            object_output = gr.Textbox(
                label="🎯 Object Detection", 
                interactive=False,
                placeholder="Detecting objects...",
                lines=4  # Increased lines
            )
            
            stats_output = gr.Textbox(
                label="📊 System Statistics",
                interactive=False,
                placeholder="Loading AI models...",
                lines=8  # Increased lines
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ## 🚀 Features:
            - **Dual AI Processing**: Gesture + Object recognition simultaneously
            - **Real-time Performance**: Optimized for web browsers  
            - **Smart Detection**: MediaPipe hands + YOLOv8 objects
            - **Live Statistics**: Model status and performance metrics
            
            ## 🎭 Gesture Support:
            👍 Thumbs Up • ✌️ Peace • ✋ Open Palm • ✊ Fist • 👌 OK Sign
            
            ## 🎯 Object Detection:
            Detects 80+ objects including: person, car, phone, laptop, cup, etc.
            """)
        
        with gr.Column():
            gr.Markdown("""
            ## 💡 Usage Tips:
            - **Good Lighting**: Improves both gesture and object detection
            - **Clear Background**: Better object separation  
            - **Steady Hands**: More accurate gesture recognition
            - **Camera Position**: Keep objects and hands in view
            
            ## ⚡ Performance:
            - **8 FPS**: Optimized for combined AI processing
            - **Auto-scaling**: Resizes frames for better performance
            - **Smart Caching**: Reduces processing load
            - **Error Handling**: Graceful fallbacks if models fail
            """)
    
    # Connect the streaming
    webcam_combined.stream(
        fn=detect_gesture_and_objects_gradio,
        inputs=webcam_combined,
        outputs=[webcam_combined, gesture_output, object_output, stats_output],
        show_progress=False
    )

def run_advanced_detection_system():
    """
    Run the advanced detection system with live camera feed and multiple models
    """
    if not ADVANCED_DETECTION_AVAILABLE:
        print("❌ Advanced detection system not available.")
        print("   Please ensure advanced_detection_system.py is present and dependencies are installed.")
        return
    
    print("\n" + "="*70)
    print("🚀 ADVANCED MULTI-MODEL OBJECT DETECTION SYSTEM (15 FPS OPTIMIZED)")
    print("="*70)
    print("🎯 Features:")
    print("  • Multiple YOLOv8 models (nano, small, medium, large, xlarge)")
    print("  • Ensemble detection for improved accuracy")
    print("  • Real-time object tracking at 15 FPS")
    print("  • Advanced statistics and performance metrics")
    print("  • Color-coded bounding boxes with confidence scores")
    print("  • 80+ object classes supported")
    print("  • Performance optimizations for smooth real-time detection")
    print("\n📋 Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save current frame")
    print("  • Press 'r' to reset statistics")
    print("  • Press 'e' to toggle ensemble mode")
    print("  • Press 'f' to cycle performance modes (Normal/Fast/Ultra)")
    print("  • Press 'c' to cycle confidence thresholds (0.4/0.5/0.6)")
    print("="*70)
    
    # Initialize the advanced detector with performance optimizations
    try:
        detector = get_advanced_detector()
        print("\n🔄 Initializing advanced detection system...")
        
        # Set optimized parameters for 15 FPS performance
        detector.confidence_threshold = 0.4  # Higher confidence for fewer detections
        detector.nms_threshold = 0.5  # Slightly higher for better NMS performance
        
        # Performance optimization: Use single model by default for 15+ FPS
        ensemble_mode = False  # Start with single model for better performance
        
        print(f"⚡ Performance mode: Single model (for 15 FPS target)")
        print(f"💡 Press 'E' during detection to enable ensemble mode if needed")
        
        if not detector.models:
            print("❌ No models could be loaded. Please run setup_enhanced_detection.py first.")
            return
        
        print(f"\n🎯 Successfully loaded {len(detector.models)} model(s) - Optimized for 15 FPS")
        print("📋 Available models:")
        for model_name, model_info in detector.models.items():
            print(f"   • {model_name} ({model_info['type']}) - {model_info['classes']} classes")
        
        # Additional optimization: Prioritize fastest model
        fastest_model = None
        for model_name, model_info in detector.models.items():
            if 'n.pt' in model_name:  # nano model is fastest
                fastest_model = model_name
                break
            elif 's.pt' in model_name:  # small model is second fastest
                fastest_model = model_name
        
        if fastest_model:
            print(f"🚀 Prioritizing fastest model: {fastest_model} for optimal performance")
        
        # Initialize camera with optimized settings for 15 FPS
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set camera properties optimized for 15 FPS performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optimized resolution for 15 FPS
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Optimized resolution for 15 FPS  
        cap.set(cv2.CAP_PROP_FPS, 15)  # Target 15 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("🎥 Camera initialized successfully")
        print("🚀 Starting advanced detection... (press 'q' to quit)")
        print("⚡ FAST MODE ENABLED by default for 15+ FPS")
        print("📋 Controls: E=Ensemble | F=Toggle Fast | C=Confidence | Q=Quit\n")
        
        # ensemble_mode already set above for performance
        frame_count = 0
        start_time = time.time()
        
        # Set optimized defaults for better multi-object detection
        detector.confidence_threshold = 0.3  # Lower confidence for more objects
        
        # Performance optimization variables for 15 FPS
        target_fps = 15
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        skip_frame_counter = 0
        process_every_nth_frame = 1  # Process every frame but with optimizations
        fast_mode = True  # Start in fast mode for 15+ FPS by default
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # FPS control - maintain 15 FPS
            if current_time - last_frame_time < frame_time:
                continue
            last_frame_time = current_time
            
            # Skip frames for performance optimization
            skip_frame_counter += 1
            if skip_frame_counter % process_every_nth_frame != 0:
                # Show previous frame if available
                if 'processed_frame' in locals():
                    cv2.imshow('Advanced Object Detection System', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                continue
            
            try:
                # Resize frame for faster processing if needed
                height, width = frame.shape[:2]
                if width > 640:
                    scale_factor = 640 / width
                    new_width = 640
                    new_height = int(height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Process frame with advanced detection (optimized for 15 FPS)
                start_detection = time.time()
                processed_frame, detections = detector.process_frame_with_tracking(
                    frame, use_ensemble=ensemble_mode, draw_stats=True
                )
                detection_time = time.time() - start_detection
                
                # Add additional information overlay (optimized)
                elapsed_time = current_time - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                detection_fps = 1.0 / detection_time if detection_time > 0 else 0
                
                # Get detector statistics (cached for performance)
                if frame_count % 15 == 0:  # Update stats every 15 frames for better performance
                    stats = detector.get_statistics()
                elif 'stats' not in locals():
                    stats = {'models_loaded': len(detector.models)}
                
                # Add status information (simplified for performance)
                mode_text = "ENSEMBLE" if ensemble_mode else "SINGLE"
                if fast_mode:
                    mode_text += " + FAST"
                
                status_text = [
                    f"🚀 Advanced Detection @ 15 FPS (Optimized)",
                    f"📊 Mode: {mode_text} | Objects: {len(detections)}",
                    f"📈 Camera FPS: {fps:.1f} | Detection: {detection_fps:.1f}",
                    f"📋 Q=Quit | E=Ensemble | F=Fast | C=Confidence | S=Save"
                ]
                
                y_offset = 30
                for i, text in enumerate(status_text):
                    cv2.putText(processed_frame, text, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Advanced Object Detection System', processed_frame)
                
                # Handle keyboard input (optimized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Exiting advanced detection system...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"advanced_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset statistics
                    detector.reset_statistics()
                    frame_count = 0
                    start_time = time.time()
                    print("🔄 Statistics reset")
                elif key == ord('e'):
                    # Toggle ensemble mode
                    ensemble_mode = not ensemble_mode
                    mode_status = "ON" if ensemble_mode else "OFF"
                    print(f"🔀 Ensemble mode: {mode_status}")
                elif key == ord('f'):
                    # Toggle fast mode (aggressive frame skipping for 15+ FPS)
                    fast_mode = not fast_mode
                    if fast_mode:
                        process_every_nth_frame = 1  # Process every frame but with faster detection
                        detector.confidence_threshold = 0.6  # Higher confidence for speed
                        mode_text = "FAST MODE ON (15+ FPS target)"
                    else:
                        process_every_nth_frame = 2  # Back to normal
                        detector.confidence_threshold = 0.4  # Back to normal confidence
                        mode_text = "NORMAL MODE (Balanced)"
                    print(f"⚡ {mode_text}")
                elif key == ord('c'):
                    # Cycle confidence threshold for performance tuning
                    if detector.confidence_threshold <= 0.4:
                        detector.confidence_threshold = 0.5
                        perf_note = "Higher confidence = Better FPS"
                    elif detector.confidence_threshold <= 0.5:
                        detector.confidence_threshold = 0.6
                        perf_note = "Highest confidence = Best FPS"
                    else:
                        detector.confidence_threshold = 0.3
                        perf_note = "Lower confidence = More objects"
                    print(f"🎚️ Confidence: {detector.confidence_threshold:.1f} ({perf_note})")
                
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                continue
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics with performance info
        final_stats = detector.get_statistics()
        print("\n" + "="*50)
        print("📊 FINAL STATISTICS (15 FPS OPTIMIZED)")
        print("="*50)
        print(f"🎯 Total detections: {final_stats['total_detections']}")
        print(f"📈 Average FPS: {final_stats['average_fps']:.1f}")
        print(f"🤖 Models used: {final_stats['models_loaded']}")
        print(f"📋 Model names: {', '.join(final_stats['model_names'])}")
        print(f"⚡ Final performance mode: {'FAST' if fast_mode else 'NORMAL'}")
        print(f"🎚️ Final confidence threshold: {detector.confidence_threshold}")
        print(f"🎥 Processing interval: Every {process_every_nth_frame} frame(s)")
        print(f"🚀 Ensemble mode used: {'Yes' if ensemble_mode else 'No (Single model for speed)'}")
        print(f"🎭 Ensemble mode: {'ON' if ensemble_mode else 'OFF'}")
        print(f"🔧 Confidence threshold: {detector.confidence_threshold}")
        if final_stats['objects_detected']:
            # Get top 5 most detected objects
            top_objects = sorted(final_stats['objects_detected'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            top_objects_str = ', '.join([f"{obj}({count})" for obj, count in top_objects])
            print(f"🔝 Most detected objects: {top_objects_str}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error initializing advanced detection system: {e}")
        print("💡 Try running setup_enhanced_detection.py to ensure proper setup")
    
    finally:
        # Ensure cleanup
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

def run_ultra_accurate_detection_system():
    """
    Run the ultra-accurate detection system for maximum precision (no FPS constraint)
    """
    if not ULTRA_ACCURATE_DETECTION_AVAILABLE:
        print("❌ Ultra-accurate detection system not available.")
        print("   Please ensure ultra_accurate_detection.py is present and dependencies are installed.")
        return
    
    print("\n" + "="*80)
    print("🎯 ULTRA-HIGH ACCURACY MULTI-OBJECT DETECTION SYSTEM")
    print("="*80)
    print("⚠️  WARNING: This mode prioritizes MAXIMUM ACCURACY over speed!")
    print("    Expect lower FPS but highest possible detection precision.")
    print("\n🎯 Features:")
    print("  • ALL YOLOv8 models (nano, small, medium, large, xlarge)")
    print("  • Advanced ensemble fusion with weighted voting")
    print("  • Clustering-based NMS for overlapping objects")
    print("  • Ultra-low confidence threshold (0.1) for maximum sensitivity")
    print("  • Detailed accuracy statistics and confidence analysis")
    print("  • 80+ object classes with enhanced multi-object detection")
    print("  • Advanced object tracking and temporal consistency")
    print("\n📋 Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save current frame with detections")
    print("  • Press 'r' to reset statistics")
    print("  • Press 'a' to toggle accuracy analysis overlay")
    print("  • Press 'c' to cycle confidence visualization modes")
    print("  • Press 't' to toggle temporal smoothing")
    print("="*80)
    
    # Ask for confirmation since this mode is resource-intensive
    confirm = input("\nThis mode uses significant computational resources. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Ultra-accurate detection cancelled.")
        return
    
    try:
        # Initialize the ultra-accurate detector
        detector = UltraAccurateDetectionSystem()
        print("\n🔄 Initializing ultra-accurate detection system...")
        
        # Load all models for maximum accuracy
        detector.load_all_models()
        
        if not detector.models:
            print("❌ No models could be loaded. Please run setup_enhanced_detection.py first.")
            return
        
        print(f"✅ Loaded {len(detector.models)} models for ultra-accurate detection")
        print("🚀 Starting ultra-accurate detection with live camera...")
        
        # Initialize camera with highest quality settings
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set high resolution for maximum accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # High FPS for smooth capture
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        detector.start_time = start_time  # Track session start
        fps_history = deque(maxlen=30)
        
        print("📺 Ultra-accurate detection window opened")
        print("⏱️  Note: Processing may be slower due to ultra-high accuracy requirements")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_start = time.time()
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run ultra-accurate detection
            detections, accuracy_stats = detector.detect_ultra_accurate(frame)
            
            # Draw all detections with enhanced visualization
            annotated_frame = detector.draw_ultra_accurate_detections_with_stats(
                frame, detections, accuracy_stats
            )
            
            # Calculate and display FPS
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Add ultra-accurate mode indicators
            cv2.putText(annotated_frame, "ULTRA-ACCURATE MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Models: {len(detector.models)}", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Display accuracy statistics if available
            if accuracy_stats:
                y_offset = 160
                for stat_name, stat_value in accuracy_stats.items():
                    if isinstance(stat_value, (int, float)):
                        cv2.putText(annotated_frame, f"{stat_name}: {stat_value:.2f}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 25
            
            # Show the ultra-accurate detection result
            cv2.imshow("Ultra-Accurate Multi-Object Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ultra_accurate_detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Saved detection result: {filename}")
            elif key == ord('r'):
                # Reset statistics
                detector.reset_stats()
                print("🔄 Statistics reset")
            elif key == ord('a'):
                # Toggle accuracy analysis
                detector.toggle_accuracy_analysis()
                print("📊 Toggled accuracy analysis overlay")
            elif key == ord('c'):
                # Cycle confidence visualization
                detector.cycle_confidence_mode()
                print("🎯 Cycled confidence visualization mode")
            elif key == ord('t'):
                # Toggle temporal smoothing
                detector.toggle_temporal_smoothing()
                print("⏱️  Toggled temporal smoothing")
            
            frame_count += 1
            
            # Print periodic statistics
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"📊 Processed {frame_count} frames in {elapsed_time:.1f}s (Avg FPS: {frame_count/elapsed_time:.2f})")
                
                # Print detection statistics
                total_detections = detector.get_total_detections()
                print(f"🎯 Total detections: {total_detections}")
                
                # Print top detected objects
                top_objects = detector.get_top_detected_objects(5)
                if top_objects:
                    print("🏆 Top detected objects:")
                    for obj, count in top_objects:
                        print(f"   {obj}: {count}")
        
    except Exception as e:
        print(f"❌ Error in ultra-accurate detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Ultra-accurate detection system stopped")
            
            # Print final statistics
            if 'detector' in locals():
                final_stats = detector.get_final_statistics()
                print("\n📊 Final Ultra-Accurate Detection Statistics:")
                for stat_name, stat_value in final_stats.items():
                    print(f"   {stat_name}: {stat_value}")
        except:
            pass

def demo_advanced_detection():
    """
    Quick demo of the advanced detection system with pre-recorded test
    """
    if not ADVANCED_DETECTION_AVAILABLE:
        print("❌ Advanced detection system not available.")
        return
    
    print("\n🎬 ADVANCED DETECTION DEMO")
    print("="*50)
    print("🚀 This will test the advanced detection system")
    print("📸 Testing with synthetic frames...")
    
    try:
        detector = get_advanced_detector()
        
        if not detector.models:
            print("❌ No models loaded. Please run setup_enhanced_detection.py first.")
            return
        
        # Create a few test frames with different patterns
        test_frames = []
        
        # Frame 1: Random noise
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Frame 2: Simple shapes (might be detected as objects)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(frame2, (400, 300), 50, (128, 128, 128), -1)
        
        # Frame 3: Text pattern
        frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame3, "TEST DETECTION", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        test_frames = [frame1, frame2, frame3]
        
        print(f"🧪 Testing {len(test_frames)} frames...")
        
        for i, frame in enumerate(test_frames):
            print(f"\n📋 Processing test frame {i+1}/{len(test_frames)}...")
            
            start_time = time.time()
            processed_frame, detections = detector.process_frame_with_tracking(frame)
            process_time = time.time() - start_time
            
            print(f"   ⏱️ Processing time: {process_time:.3f}s")
            print(f"   🎯 Objects detected: {len(detections)}")
            
            if detections:
                for j, det in enumerate(detections[:3]):  # Show first 3 detections
                    print(f"     {j+1}. {det['class_name']} (confidence: {det['confidence']:.2f})")
        
        # Show final statistics
        stats = detector.get_statistics()
        print(f"\n📊 DEMO STATISTICS:")
        print(f"   🤖 Models loaded: {stats['models_loaded']}")
        print(f"   📈 Average FPS: {stats['average_fps']:.1f}")
        print(f"   🎯 Total detections: {stats['total_detections']}")
        
        print("\n✅ Demo completed successfully!")
        print("💡 To run with live camera, use option 11 in the main menu")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Try running setup_enhanced_detection.py to ensure proper setup")

# Extended menu with combined Gradio option
def run_with_combined_gradio_options():
    """Extended main function with combined Gradio interface"""
    print("\n" + "="*60)
    print("🤖 AI VISION SYSTEM - Extended Options")
    print("="*60)
    
    choice = input("""
Choose option:
1. Run complete gesture pipeline
2. Collect gesture data only
3. Train gesture model only
4. Start gesture recognition only
5. Test camera only
6. Use OpenCV gesture interface (RECOMMENDED)
7. 🆕 Object Detection Only
8. 🆕 Combined Gesture + Object Detection (OpenCV)
9. Gradio web interface (gesture only)
10. 🚀 Combined AI Vision (Gradio Web Interface)
11. 🎯 ADVANCED Multi-Model Detection System (OPTIMIZED FOR 15 FPS!)
12. 🎬 Advanced Detection Demo (No Camera Required)
13. 🧪 Test 15 FPS Performance (Benchmark Tool)
14. 🏆 ULTRA-ACCURATE Multi-Object Detection (MAXIMUM PRECISION!)

Enter (1-14): """)
    
    if choice == "10":
        print("\n🚀 Starting Combined AI Vision Web Interface...")
        print("📱 This includes both gesture recognition AND object detection")
        print("🌐 Opening in browser: http://localhost:7860")
        print("📺 FULL SCREEN MODE - Enhanced for better multi-object detection")
        print("⚡ Optimized for web performance with dual AI processing")
        combined_iface.launch(
            server_name="127.0.0.1", 
            server_port=7860, 
            share=False, 
            inbrowser=True,
            show_error=True,
            favicon_path=None,
            app_kwargs={"docs_url": None}  # Clean interface
        )
    elif choice == "11":
        # Advanced Multi-Model Detection System (15 FPS Optimized)
        print("\n🎯 Starting 15 FPS Optimized Advanced Detection System...")
        print("⚡ Performance Features:")
        print("   • Intelligent frame processing (every 2nd frame)")
        print("   • Single model mode by default for 15+ FPS")
        print("   • Optimized confidence thresholds for speed")
        print("   • Fast mode available (F key)")
        print("   • Real-time FPS monitoring")
        print("   • Cached statistics for better performance")
        print("🎮 Controls:")
        print("   • 'E' - Toggle ensemble mode (slower but more accurate)")
        print("   • 'F' - Toggle fast mode (15+ FPS target)")
        print("   • 'C' - Adjust confidence threshold")
        print("   • 'Q' - Quit")
        run_advanced_detection_system()
    elif choice == "12":
        # Advanced Detection Demo
        demo_advanced_detection()
    elif choice == "13":
        # Test 15 FPS Performance
        print("\n🧪 Starting 15 FPS Performance Test...")
        print("📊 This will benchmark the advanced detection system")
        try:
            import subprocess
            subprocess.run([sys.executable, "test_15fps_performance.py"])
        except Exception as e:
            print(f"❌ Could not run performance test: {e}")
            print("💡 You can run it manually: python test_15fps_performance.py")
    elif choice == "14":
        # Ultra-Accurate Multi-Object Detection
        print("\n🏆 Starting Ultra-Accurate Multi-Object Detection System...")
        print("⚠️  WARNING: This mode prioritizes maximum accuracy over speed")
        print("🎯 Features: All YOLO models, advanced ensemble, ultra-low thresholds")
        run_ultra_accurate_detection_system()
    # ...existing code for other options...
    elif choice == "7":
        print("\n🎯 Starting Object Detection Only...")
        detect_objects_only_opencv()
    elif choice == "8":
        print("\n🤖 Starting Combined AI Vision (OpenCV)...")
        detect_gesture_and_objects_opencv()
    elif choice == "9":
        print("🚀 Starting Gradio web interface (gesture only)...")
        print("📱 Open in browser: http://localhost:7860")
        iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    elif choice == "1":
        if run_complete_pipeline():
            print("🚀 Starting optimized Gradio interface...")
            print("📱 Open in browser: http://localhost:7860")
            print("⚡ Note: For best performance, use option 6 (OpenCV)")
            iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    elif choice == "2":
        collect_gesture_data_integrated()
    elif choice == "3":
        train_model_advanced()
    elif choice == "4":
        print("🚀 Starting optimized Gradio interface...")
        print("📱 Open in browser: http://localhost:7860")
        print("⚡ Note: For best performance, use option 6 (OpenCV)")
        iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
    elif choice == "5":
        # Test camera function
        print("🔍 Testing camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is working!")
            ret, frame = cap.read()
            if ret:
                print("✅ Can capture frames!")
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            else:
                print("❌ Cannot capture frames!")
        else:
            print("❌ Cannot open camera!")
        cap.release()
    elif choice == "6":
        # Use OpenCV interface instead of Gradio
        print("🚀 Starting OpenCV interface...")
        predict_gesture_realtime()
    else:
        print("🚀 Starting optimized Gradio interface...")
        print("📱 Open in browser: http://localhost:7860")
        print("⚡ Note: For best performance, use option 6 (OpenCV)")
        iface.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

def run_main_menu():
    """Display the main menu for the hybrid detection system"""
    print("\n" + "="*60)
    print("🚀 HYBRID ULTIMATE DETECTION SYSTEM - MAIN MENU")
    print("="*60)
    print("1. Run Hybrid Ultimate Detection (Fullscreen Mode)")
    print("2. Launch Web Interface")
    print("3. Run with Custom Options")
    print("4. Train Gesture Model")
    print("5. Exit")
    print("="*60)
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        # Run the fullscreen detection system
        run_hybrid_ultimate_detection_system()
    elif choice == '2':
        # Launch the web interface
        try:
            from hybrid_web_interface import create_web_interface
            create_web_interface()
        except ImportError:
            print("❌ Web interface not available. Make sure hybrid_web_interface.py exists and dependencies are installed.")
    elif choice == '3':
        # Run with custom options using the launcher
        try:
            import launch_hybrid
            launch_hybrid.main()
        except ImportError:
            print("❌ Launcher not available. Make sure launch_hybrid.py exists.")
    elif choice == '4':
        # Train gesture model
        if os.path.exists("gesture_data.csv"):
            train_gesture_model()
        else:
            print("❌ No gesture data found. Please collect gesture data first.")
    elif choice == '5':
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please try again.")
    
    # Return to the menu
    run_main_menu()

def train_gesture_model():
    """Train and save a gesture recognition model from collected data"""
    try:
        print("🧠 Training gesture recognition model...")
        
        # Load the dataset
        data = pd.read_csv("gesture_data.csv")
        print(f"📊 Loaded {len(data)} gesture samples")
        
        if len(data) < 20:
            print("⚠️ Not enough gesture samples (min 20 required). Please collect more data.")
            return
        
        # Split features and target
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Create a model pipeline with feature selection and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=40)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train with cross-validation
        print("🔄 Training model with cross-validation...")
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        print(f"📈 Cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model on all data
        pipeline.fit(X, y)
        
        # Save the model
        joblib.dump(pipeline, "gesture_model.pkl")
        print("✅ Model trained and saved to gesture_model.pkl")
        
    except Exception as e:
        print(f"❌ Error training gesture model: {str(e)}")

if __name__ == "__main__":
    if HYBRID_DETECTION_AVAILABLE:
        run_main_menu()
    else:
        print("❌ Hybrid Ultimate Detection System not available.")
        print("Please make sure hybrid_ultimate_detection.py is in the current directory and dependencies are installed.")
        print("You can install dependencies by running: pip install -r requirements.txt")