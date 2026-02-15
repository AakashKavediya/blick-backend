import cv2
import pyautogui
import time
import mediapipe as mp
import math

def run_smile_control(cap, stop_flag, calibration_manager):
    """Smile detection using lip corner distance (MediaPipe)"""
    
    # Disable PyAutoGUI failsafe for smooth operation
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.05  # Small pause between pyautogui commands
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_smile_time = 0
    smile_cooldown = 2.0
    
    # Get calibration
    calibration = calibration_manager.get_smile_calibration()
    if not calibration:
        print("⚠️ No calibration found! Using defaults.")
        neutral_distance = 50.0
        smile_threshold = 10.0
    else:
        neutral_distance = calibration.get('neutral_intensity', 50.0)
        smile_threshold = calibration.get('smile_threshold', 10.0)
    
    trigger_distance = neutral_distance + smile_threshold
    
    print("\n" + "="*50)
    print("😊 SMILE DETECTION ACTIVE (Lip Corner Distance)")
    print("="*50)
    print(f"✅ Neutral distance: {neutral_distance:.1f}")
    print(f"✅ Trigger when distance > {trigger_distance:.1f}")
    print("✅ Press Q to quit")
    print("="*50 + "\n")
    
    cv2.namedWindow('Smile Control - Q to Quit', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Smile Control - Q to Quit', 100, 100)
    cv2.resizeWindow('Smile Control - Q to Quit', 640, 480)
    
    # MediaPipe landmarks for lip corners
    LEFT_LIP_CORNER = 61   # Left corner of mouth
    RIGHT_LIP_CORNER = 291  # Right corner of mouth
    
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        h, w, _ = frame.shape
        status = "No face detected"
        color = (0, 0, 255)
        current_distance = 0.0
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get left and right lip corner coordinates
            left_corner = landmarks[LEFT_LIP_CORNER]
            right_corner = landmarks[RIGHT_LIP_CORNER]
            
            x1 = int(left_corner.x * w)
            y1 = int(left_corner.y * h)
            x2 = int(right_corner.x * w)
            y2 = int(right_corner.y * h)
            
            # Calculate distance between lip corners
            current_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Draw lip corners
            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Display values
            cv2.putText(frame, f"Distance: {current_distance:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Need > {trigger_distance:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            progress = min(100, (current_distance / trigger_distance) * 100)
            bar_width = int((progress / 100) * 300)
            cv2.rectangle(frame, (10, 80), (310, 100), (100, 100, 100), 2)
            cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)
            cv2.putText(frame, f"{progress:.0f}%", (320, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Smile detection
            if current_distance > trigger_distance:
                if time.time() - last_smile_time > smile_cooldown:
                    print(f"😊 SMILE DETECTED! Switching tab (distance={current_distance:.1f})")
                    
                    try:
                        # Minimize CV window to allow browser to get focus
                        cv2.setWindowProperty('Smile Control - Q to Quit', 
                                            cv2.WND_PROP_VISIBLE, 0)
                        time.sleep(0.15)  # Give time for window to minimize and browser to focus
                        
                        # Use hotkey method - more reliable than individual keys
                        pyautogui.hotkey('ctrl', 'tab')
                        
                        last_smile_time = time.time()
                        status = "✅ TAB SWITCHED!"
                        color = (0, 255, 0)
                        print("✅ Tab switched successfully!\n")
                        
                        time.sleep(0.1)
                        # Restore CV window
                        cv2.setWindowProperty('Smile Control - Q to Quit', 
                                            cv2.WND_PROP_VISIBLE, 1)
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        status = "❌ Error!"
                        color = (0, 0, 255)
                        # Make sure window is visible even if error occurs
                        try:
                            cv2.setWindowProperty('Smile Control - Q to Quit', 
                                                cv2.WND_PROP_VISIBLE, 1)
                        except:
                            pass
                else:
                    cooldown_rem = smile_cooldown - (time.time() - last_smile_time)
                    status = f"Cooldown {cooldown_rem:.1f}s"
                    color = (255, 255, 0)
            
            elif current_distance > (neutral_distance + smile_threshold * 0.5):
                status = "Smile BIGGER!"
                color = (0, 165, 255)
            else:
                status = "Neutral - SMILE!"
                color = (150, 150, 150)
        else:
            status = "No face detected"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.imshow('Smile Control - Q to Quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\n✅ Smile detection stopped\n")
